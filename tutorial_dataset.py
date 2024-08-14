import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/CAMUS/prompt_camus.json', 'rt') as f: # directory leading to the prompts
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/CAMUS/' + source_filename) # "source" is the condition
        target = cv2.imread('./training/CAMUS/' + target_filename) # "target" is the data

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Resize the images to 512x512 in order to avoid any errors
        source = cv2.resize(source, (512,512), interpolation= cv2.INTER_LINEAR) # https://learnopencv.com/image-resizing-with-opencv/#resize-by-wdith-height
        target = cv2.resize(target, (512,512), interpolation= cv2.INTER_LINEAR) # https://learnopencv.com/image-resizing-with-opencv/#resize-by-wdith-height
        
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

