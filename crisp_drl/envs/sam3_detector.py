import numpy as np
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from PIL import Image


class Sam3Detector:
    def __init__(self):
        model = build_sam3_image_model(
            #checkpoint_path="/home/gabor/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt",
        )
        processor = Sam3Processor(model, confidence_threshold=0.01)
        self.processor = processor

    def segment_lego(
        self,
        image: np.ndarray,
        colors: tuple[str, str] = ("lavender", "purple"),
    ) -> dict[str, np.ndarray]:
        """Segment two LEGO bricks using color-specific prompts.

        Args:
            image: HxWx3 RGB image.
            colors: ``(color1, color2)`` — color names to use in SAM3 prompts.
                Defaults to ``("lavender", "purple")`` for backward compat.
                Use ``("yellow", "lavender")`` for the yellow+lavender task.

        Returns:
            Dict mapping each color name to its binary mask (uint8, 0/255).
        """
        inference_state = self.processor.set_image(Image.fromarray(image))
        result = {}

        # Query SAM3 for each color separately using color-specific prompts
        for color_name in colors:
            output = self.processor.set_text_prompt(
                state=inference_state, prompt=f"small {color_name} lego brick"
            )
            masks, _boxes, scores = output["masks"], output["boxes"], output["scores"]
            scores = scores.cpu().numpy()

            if len(masks) == 0:
                print(f"[SAM3] Warning: No mask found for {color_name}")
                continue

            # Select the mask with highest confidence score
            best_mask_idx = np.argmax(scores)
            hw = (masks[0].shape[-2], masks[0].shape[-1])
            result[color_name] = masks[best_mask_idx].cpu().numpy().reshape(*hw) * 255
            print(f"[SAM3] Detected {color_name} brick (confidence: {scores[best_mask_idx]:.3f})")

        # Fallback: if color prompting didn't find both bricks, use generic detection
        if len(result) < 2:
            print(f"[SAM3] Color-specific prompts found only {len(result)} brick(s), falling back to generic detection")
            output = self.processor.set_text_prompt(
                state=inference_state, prompt="small lego brick"
            )
            masks, _boxes, scores = output["masks"], output["boxes"], output["scores"]
            scores = scores.cpu().numpy()

            if len(masks) > 2:
                masks = masks[np.argsort(scores)[-2:]]
                scores_sorted = np.sort(scores)
                if scores_sorted[-2] - scores_sorted[-3] < 0.2:
                    print(
                        f"Warning: more than two masks detected (scores {scores.tolist()}), "
                        "choosing the two with highest scores, although the difference to the third "
                        f"highest score is only {scores_sorted[-2] - scores_sorted[-3]:.2f}"
                    )

            # Use brightness-based assignment only as fallback
            image_region_means = []
            for mask in masks:
                masked_image = np.array(image) * mask.cpu().numpy().reshape(
                    mask.shape[-2], mask.shape[-1], 1
                )
                mean_color = masked_image.sum(axis=(0, 1)) / mask.sum().cpu().numpy()
                image_region_means.append(mean_color.mean())

            lightest_mask_idx = np.argmax(image_region_means)
            darkest_mask_idx = np.argmin(image_region_means)
            hw = (masks[0].shape[-2], masks[0].shape[-1])

            # Assign by brightness to the remaining missing colors
            missing_colors = [c for c in colors if c not in result]
            if len(missing_colors) >= 1:
                result[missing_colors[0]] = masks[lightest_mask_idx].cpu().numpy().reshape(*hw) * 255
            if len(missing_colors) >= 2:
                result[missing_colors[1]] = masks[darkest_mask_idx].cpu().numpy().reshape(*hw) * 255

        return result

    def segment_siemens(self, image: np.ndarray) -> np.ndarray:
        inference_state = self.processor.set_image(Image.fromarray(image))
        output = self.processor.set_text_prompt(
            state=inference_state, prompt="black cover with circular grille"
        )
        masks, _boxes, scores = output["masks"], output["boxes"], output["scores"]
        scores = scores.cpu().numpy()
        if len(masks) == 0:
            raise ValueError("[SAM3 Segment] No masks found")
        mask_idx = 0
        if len(masks) > 1:
            scores_sorted = np.sort(scores)
            if scores_sorted[-1] - scores_sorted[-2] < 0.1:
                print(
                    f"Warning: more than one mask detected (scores {scores.tolist()}), "
                    "choosing the one with highest score, although the difference to the second "
                    f"highest score is only {scores_sorted[-1] - scores_sorted[-2]:.2f}"
                )
            if scores_sorted[-1] < 0.5:
                print(
                    f"Warning: best mask is not confident, score: {scores_sorted[-1]:.2f} "
                )
            mask_idx = np.argsort(scores)[-1]

        return (
            masks[mask_idx]
            .cpu()
            .numpy()
            .reshape(masks[0].shape[-2], masks[0].shape[-1])
            * 255
        )
