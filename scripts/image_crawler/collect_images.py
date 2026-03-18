from pathlib import Path

from icrawler.builtin import BingImageCrawler


def crawl_images(keyword, output_dir, max_num=40):
    output_dir.mkdir(parents=True, exist_ok=True)
    crawler = BingImageCrawler(storage={"root_dir": str(output_dir)})
    crawler.crawl(keyword=keyword, max_num=max_num)


def main():
    base_dir = Path("data/images_collected")

    targets = [
        ("night signboard street korea", base_dir / "sign", 40),
        ("street light night road", base_dir / "streetlight", 40),
        ("night building lights exterior", base_dir / "light", 40),
    ]

    for keyword, out_dir, count in targets:
        print(f"[START] {keyword} -> {out_dir}")
        crawl_images(keyword, out_dir, count)
        print(f"[DONE]  {keyword}")

    print("All image crawling tasks finished.")


if __name__ == "__main__":
    main()
