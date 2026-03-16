import os
from PIL import Image
base='data/images'
os.makedirs(os.path.join(base,'train/images'),exist_ok=True)
os.makedirs(os.path.join(base,'train/labels'),exist_ok=True)
os.makedirs(os.path.join(base,'val/images'),exist_ok=True)
os.makedirs(os.path.join(base,'val/labels'),exist_ok=True)
for split in ['train','val']:
    img_path=os.path.join(base, split, 'images', f'sample_{split}.jpg')
    lbl_path=os.path.join(base, split, 'labels', f'sample_{split}.txt')
    im = Image.new('RGB', (640, 640), (20, 20, 20))
    im.save(img_path)
    with open(lbl_path, 'w', encoding='utf-8') as f:
        f.write('0 0.5 0.5 0.5 0.5\n')
print('sample dataset created')
