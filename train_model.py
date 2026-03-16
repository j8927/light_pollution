import os
import argparse
import yaml
from ultralytics import YOLO


def ensure_data_config(data_dir, out_path="data/light_pollution.yaml"):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"데이터 폴더가 올바르지 않습니다. \n준비 순서:\n 1) {data_dir}/train/images, {data_dir}/train/labels\n 2) {data_dir}/val/images, {data_dir}/val/labels"
        )
    data_cfg = {
        'path': data_dir,
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,
        'names': ['light_object']
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_cfg, f, allow_unicode=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 학습 스크립트 (빛 공해)')
    parser.add_argument('--data', default='data/images', help='데이터 루트 폴더 (기본: data/images)')
    parser.add_argument('--epochs', type=int, default=30, help='학습 epoch 수')
    parser.add_argument('--batch', type=int, default=8, help='배치 크기')
    parser.add_argument('--imgsz', type=int, default=640, help='이미지 사이즈')
    parser.add_argument('--model', default='yolov8n.pt', help='기본 YOLO 모델 체크포인트')
    parser.add_argument('--output', default='models', help='저장 폴더')
    args = parser.parse_args()

    data_cfg = os.path.join(args.data, 'light_pollution.yaml')
    if not os.path.exists(data_cfg):
        try:
            data_cfg = ensure_data_config(args.data, out_path=data_cfg)
            print(f"데이터 config 생성: {data_cfg}")
        except Exception as e:
            print(e)
            return

    os.makedirs(args.output, exist_ok=True)
    model = YOLO(args.model)
    print('학습 시작: model=', args.model, 'data=', data_cfg)
    result = model.train(
        data=data_cfg,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.output,
        name='light_pollution',
        exist_ok=True,
        plots=False,
        save=False,
        verbose=False,
    )
    import glob
    best = None
    for p in glob.glob('**/weights/best.*', recursive=True):
        if os.path.basename(p).startswith('best'):
            best = p
            break
    if not best:
        print('best 체크포인트를 찾을 수 없습니다. runs/*/weights 폴더를 확인하세요.')
        return
    print('학습 완료. 체크포인트 위치:', best)
    dest = os.path.join('models', 'light_pollution_best.pt')
    os.makedirs('models', exist_ok=True)
    import shutil
    shutil.copy(best, dest)
    print('모델 복사 완료:', dest)


if __name__ == '__main__':
    main()
