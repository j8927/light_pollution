import os
import argparse
import yaml
from ultralytics import YOLO


def ensure_data_config(data_dir, out_path="data/light_pollution.yaml"):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    valid_dir = os.path.join(data_dir, 'valid')
    eval_dir = val_dir if os.path.isdir(val_dir) else valid_dir
    if not os.path.isdir(train_dir) or not os.path.isdir(eval_dir):
        raise FileNotFoundError(
            f"데이터 폴더가 올바르지 않습니다. \n준비 순서:\n 1) {data_dir}/train/images, {data_dir}/train/labels\n 2) {data_dir}/val(images) 또는 {data_dir}/valid(images)"
        )
    # Roboflow 데이터셋 클래스 3개
    # 라벨 파일의 class_id: 0=간판, 1=조명, 2=가로등
    data_cfg = {
        'path': data_dir,
        'train': 'train/images',
        'val': 'val/images' if os.path.isdir(val_dir) else 'valid/images',
        'nc': 3,
        'names': ['간판', '조명', '가로등']
    }   
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_cfg, f, allow_unicode=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 학습 스크립트 (빛 공해)')
    parser.add_argument(
        '--data',
        default='data/images',
        help='데이터 경로(폴더 또는 data.yaml). 기본: data/images'
    )
    parser.add_argument('--epochs', type=int, default=100, help='학습 epoch 수')
    parser.add_argument('--batch', type=int, default=8, help='배치 크기')
    parser.add_argument('--imgsz', type=int, default=640, help='이미지 사이즈')
    parser.add_argument('--model', default='yolov8n.pt', help='기본 YOLO 모델 체크포인트')
    parser.add_argument('--output', default='models', help='저장 폴더')
    parser.add_argument('--name', default='light_pollution', help='학습 실험 이름')
    args = parser.parse_args()

    if os.path.isfile(args.data) and args.data.lower().endswith(('.yaml', '.yml')):
        data_cfg = args.data
    else:
        data_cfg = os.path.join(args.data, 'light_pollution.yaml')
        if not os.path.exists(data_cfg):
            try:
                data_cfg = ensure_data_config(args.data, out_path=data_cfg)
                print(f"데이터 config 생성: {data_cfg}")
            except Exception as e:
                print(e)
                return

    os.makedirs(args.output, exist_ok=True)

    import torch
    device = '0' if torch.cuda.is_available() else 'cpu'
    workers = min(8, os.cpu_count() or 4)
    print(f"장치: {device} | CPU 코어: {os.cpu_count()} | Worker: {workers}")

    model = YOLO(args.model)
    print('학습 시작: model=', args.model, 'data=', data_cfg)
    result = model.train(
        data=data_cfg,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.output,
        name=args.name,
        exist_ok=True,
        plots=False,
        save=True,
        verbose=True,
        device=device,
        workers=workers,
        cache=True,
        patience=0,
    )
    import glob
    best_candidates = [
        p for p in glob.glob('**/weights/best.*', recursive=True)
        if os.path.basename(p).startswith('best')
    ]
    best = max(best_candidates, key=os.path.getmtime) if best_candidates else None
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
