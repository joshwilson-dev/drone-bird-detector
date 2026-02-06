from torchvision.models.detection.roi_heads import RoIHeads
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms
from torchvision.ops.boxes import clip_boxes_to_image

class MultiClassRoIHeads(RoIHeads):
    def postprocess_detections_multiclass(
        self,
        class_logits,
        box_regression,
        proposals,
        image_shapes,
    ):
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(p) for p in proposals]

        # Decode boxes
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, dim=-1)
        bg_scores = pred_scores[:, 0]
        fg_scores = pred_scores[:, 1:]

        # Split per image
        pred_boxes = pred_boxes.split(boxes_per_image, dim=0)
        bg_scores = bg_scores.split(boxes_per_image, dim=0)
        fg_scores = fg_scores.split(boxes_per_image, dim=0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_fg_scores = []
        all_bg_scores = []

        for boxes_i, fg_scores_i, bg_score_i, image_shape in zip(pred_boxes, fg_scores, bg_scores, image_shapes):

            # boxes: [N, C - 1, 4], scores: [N, C - 1]
            boxes = boxes_i.reshape(-1, num_classes, 4)[:, 1:]
            scores = fg_scores_i.reshape(-1, num_classes -1)
            bg = bg_score_i

            num_classes_no_bg = scores.shape[1]

            # Create class labels
            proposal_ids = torch.arange(
                boxes.shape[0], device=device
            ).view(-1, 1).expand_as(scores)
            
            class_ids = torch.arange(
                num_classes_no_bg, device=device
            ).view(1, -1).expand_as(scores)

            # Flatten
            boxes = boxes.reshape(-1, 4)
            scores_flat = scores.reshape(-1)
            labels = class_ids.reshape(-1)
            proposal_ids = proposal_ids.reshape(-1)

            # Threshold
            keep = scores_flat > self.score_thresh
            boxes = boxes[keep]
            scores_flat = scores_flat[keep]
            labels = labels[keep]
            proposal_ids = proposal_ids[keep]

            # Batched NMS
            keep_idx = batched_nms(
                boxes,
                scores_flat,
                labels,
                self.nms_thresh,
            )
            keep_idx = keep_idx[: self.detections_per_img]

            boxes = boxes[keep_idx]
            scores_flat = scores_flat[keep_idx]
            labels = labels[keep_idx]

            if boxes.numel() == 0:
                all_boxes.append(boxes)
                all_scores.append(scores_flat[:0])
                all_labels.append(labels[:0] + 1)
                all_fg_scores.append(fg_scores_i[:0])
                all_bg_scores.append(bg[:0])
                continue

            # Recover proposal index for each kept detection
            proposal_idx = proposal_ids[keep_idx].unique()
            class_scores = fg_scores_i[proposal_idx]
            bg_class_scores = bg[proposal_idx]

            boxes = clip_boxes_to_image(boxes, image_shape)

            max_scores, max_labels = class_scores.max(dim=1)

            all_boxes.append(boxes)
            all_scores.append(max_scores)
            all_labels.append(max_labels + 1)
            all_fg_scores.append(class_scores)
            all_bg_scores.append(bg_class_scores)

        return all_boxes, all_scores, all_labels, all_fg_scores, all_bg_scores

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            return super().forward(features, proposals, image_shapes, targets)

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        boxes, scores, labels, all_fg_scores, all_bg_scores = self.postprocess_detections_multiclass(
            class_logits,
            box_regression,
            proposals,
            image_shapes,
        )

        results = []
        for i in range(len(boxes)):
            results.append({
                "boxes": boxes[i],
                "scores": scores[i],
                "labels": labels[i],
                "all_fg_scores": all_fg_scores[i],
                "bg_scores": all_bg_scores[i]
            })

        return results, {}