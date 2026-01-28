import cv2
import numpy as np
import torch
import math
from typing import List, Tuple, Union

class ImageProcessor:
    def __init__(self, ar_strategy: str = 'crop'):
        """
        Args:
            ar_strategy: 'crop', 'warp', or 'tile' for handling non-square images.
        """
        self.ar_strategy = ar_strategy
        # Standard ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def load_and_normalize(self, path: str) -> np.ndarray:
        """
        Reads 16-bit Grayscale TIFF, normalizes to 0-1, converts to RGB.
        Returns float32 HWC image.
        """
        # Read 16-bit image
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError(f"Failed to read image: {path}")

        # Ensure 16-bit; if 8-bit or other, normalize accordingly
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        elif img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            # Fallback for other types, assume float or max value based on dtype
            if np.issubdtype(img.dtype, np.floating):
                pass # Already float, assume 0-1 or check max
            else:
                # fallback
                img = img.astype(np.float32) / np.iinfo(img.dtype).max

        # Convert Grayscale to RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            pass # Already RGB
        else:
            # Handle Alpha channel or other cases if necessary
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        return img

    def get_target_size(self, type_label: str) -> int:
        if "small" in type_label or "medium" in type_label:
            return 256
        elif "large" in type_label:
            return 512
        elif "huge" in type_label:
            return 512 # Patch size
        return 256 # Default

    def process(self, path: str, type_label: str) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Main entry point.
        Returns:
            Single Tensor (1, 3, H, W) for standard/small/medium/large
            List of Tensors [N, 3, 512, 512] for huge or tiled strategy
            (Actually, we return batch dimension for consistency, so (1,3,H,W) or list of (1,3,H,W))
        """
        img = self.load_and_normalize(path)
        
        is_huge = "huge" in type_label
        is_square = "square" in type_label
        target_size = self.get_target_size(type_label)

        if is_huge:
            return self._process_huge(img)
        
        if is_square:
            return self._process_simple_resize(img, target_size)
        
        # Non-square handling
        if self.ar_strategy == 'crop':
            return self._process_crop(img, target_size)
        elif self.ar_strategy == 'warp':
            return self._process_simple_resize(img, target_size) # Warp is just force resize
        elif self.ar_strategy == 'tile':
            return self._process_tile(img, target_size)
        else:
            # Fallback to crop
            return self._process_crop(img, target_size)

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Applies normalization and converts to Tensor (C, H, W)."""
        # Normalize (img is 0-1 RGB)
        img = (img - self.mean) / self.std
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).unsqueeze(0).float() # Add batch dim

    def _process_simple_resize(self, img: np.ndarray, target_size: int) -> torch.Tensor:
        """Resizes image to target_size x target_size (warping if not square)."""
        resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        return self._to_tensor(resized)

    def _process_crop(self, img: np.ndarray, target_size: int) -> torch.Tensor:
        """Aspect Ratio Preserving Resize (short edge = target) + Center Crop."""
        h, w = img.shape[:2]
        scale = target_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Center Crop
        start_x = (new_w - target_size) // 2
        start_y = (new_h - target_size) // 2
        cropped = resized[start_y:start_y+target_size, start_x:start_x+target_size]
        return self._to_tensor(cropped)

    def _process_tile(self, img: np.ndarray, target_size: int) -> List[torch.Tensor]:
        """Resize long edge, tile/pad multiple crops."""
        # Implementation choice: Resize long edge to nearest multiple of target_size? 
        # Or just resize such that short edge >= target_size?
        # The prompt says: "Resize long edge, tile/pad multiple crops".
        # A common approach: Resize short edge to target_size (preserving AR). 
        # Then tile along the long edge with overlap or padding.
        
        # However, "Resize long edge" suggests we want to cover the whole image.
        # Let's implement: Resize so short edge = target_size. Then slide window.
        h, w = img.shape[:2]
        scale = target_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        crops = []
        # Simple sliding window with stride = target_size (no overlap for simplicity unless needed)
        # If we want full coverage, we might need overlap. 
        # Let's do strict tiling with padding for the last tile.
        
        for y in range(0, new_h, target_size):
            for x in range(0, new_w, target_size):
                # Check if this is a partial tile
                patch = np.zeros((target_size, target_size, 3), dtype=np.float32)
                
                end_y = min(y + target_size, new_h)
                end_x = min(x + target_size, new_w)
                
                h_slice = end_y - y
                w_slice = end_x - x
                
                patch[:h_slice, :w_slice] = resized[y:end_y, x:end_x]
                crops.append(self._to_tensor(patch))
        
        return crops

    def _process_huge(self, img: np.ndarray) -> List[torch.Tensor]:
        """
        Huge Pipeline:
        Resize standard huge image to an intermediate resolution (e.g., long edge 2048).
        Grid crop into multiple 512x512 patches.
        """
        INTERMEDIATE_LONG_EDGE = 2048
        PATCH_SIZE = 512
        
        h, w = img.shape[:2]
        scale = INTERMEDIATE_LONG_EDGE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        patches = []
        # Grid crop 512x512 with padding if necessary
        # Note: If the resized image is smaller than 512 (unlikely for 'huge'), pad it.
        
        # Ensure at least one patch
        if new_h < PATCH_SIZE or new_w < PATCH_SIZE:
             # Pad to 512x512
             padded = np.zeros((max(new_h, PATCH_SIZE), max(new_w, PATCH_SIZE), 3), dtype=np.float32)
             padded[:new_h, :new_w] = resized
             return [self._to_tensor(padded[:PATCH_SIZE, :PATCH_SIZE])]

        # Sliding window
        stride = PATCH_SIZE # Non-overlapping? "Grid crop" usually implies non-overlapping.
        
        for y in range(0, new_h, stride):
            for x in range(0, new_w, stride):
                patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)
                
                end_y = min(y + PATCH_SIZE, new_h)
                end_x = min(x + PATCH_SIZE, new_w)
                
                h_slice = end_y - y
                w_slice = end_x - x
                
                patch[:h_slice, :w_slice] = resized[y:end_y, x:end_x]
                patches.append(self._to_tensor(patch))
                
        return patches

