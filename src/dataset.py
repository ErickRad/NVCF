import os
import glob
import cv2
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class VideoFramesDataset(Dataset):
    def __init__(self, root_dir, frames_per_clip=5, step_between_clips=5, transform=None):
        self.root_dir = root_dir
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.transform = transform
        self.samples = []
        self._extract_frames_if_needed()
        self._prepare_clips()

    def _extract_frames_if_needed(self):
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            for file_name in os.listdir(class_path):
                if not file_name.endswith(".avi"):
                    continue

                video_path = os.path.join(class_path, file_name)
                base_name = os.path.splitext(file_name)[0]
                output_folder = os.path.join(class_path, base_name)

                if os.path.exists(output_folder) and len(os.listdir(output_folder)) > 0:
                    continue  # Already extracted

                os.makedirs(output_folder, exist_ok=True)

                cap = cv2.VideoCapture(video_path)
                frame_id = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_path = os.path.join(output_folder, f"frame_{frame_id:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_id += 1
                cap.release()

    def _prepare_clips(self):
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            for video_folder in os.listdir(class_path):
                video_path = os.path.join(class_path, video_folder)
                if not os.path.isdir(video_path):
                    continue

                frames = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
                total = len(frames)

                for i in range(0, total - self.frames_per_clip + 1, self.step_between_clips):
                    clip = frames[i:i + self.frames_per_clip]
                    if len(clip) == self.frames_per_clip:
                        self.samples.append(clip)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_paths = self.samples[idx]
        clip = [Image.open(p).convert('RGB') for p in clip_paths]
        if self.transform:
            clip = [self.transform(img) for img in clip]
        clip = torch.stack(clip)  # [T, C, H, W]
        return clip

class Loader:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def load(self):
        dataset = VideoFramesDataset(
            root_dir="./data/UCF-101",
            frames_per_clip=5,
            step_between_clips=5,
            transform=self.transform
        )

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

        return train_loader, test_loader
