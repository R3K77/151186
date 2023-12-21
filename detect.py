import json
from pathlib import Path
from typing import Dict
import click
from tqdm import tqdm
import yolov7

model = yolov7.load('151186_model_wykrywanie_lisci.pt')

# set model parameters
model.conf = 0.75  # NMS confidence threshold
model.iou = 0.35  # NMS IoU threshold
model.classes = None  # (optional list) filter by class


def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        A dictionary with the number of each object.
    """
    # Wyzerowanie wartości wykrytych liści
    aspen = 0
    birch = 0
    hazel = 0
    maple = 0
    oak = 0

    # Wyniki detekcji liści na obrazie
    results = model(img_path)

    # Zapisanie wyników detekcji do zmiennych
    predictions = results.pred[0]
    # boxes = predictions[:, :4] # x1, y1, x2, y2
    # scores = predictions[:, 4]
    categories = predictions[:, 5]  # 0, 1, 2, 3, 4 = aspen, birch, hazel, maple, oak

    # Zliczanie ilości wykrytych liści
    for category in categories:
        category = int(category)
        if category == 0:
            aspen += 1
        elif category == 1:
            birch += 1
        elif category == 2:
            hazel += 1
        elif category == 3:
            maple += 1
        elif category == 4:
            oak += 1

    # Otwarcie obrazu z zaznaczonymi wykrytymi liśćmi
    ##results.show()

    # Zapisanie wyników do folderu WYNIKI jako plik .jpg
    results.save(save_dir='WYNIKI')

    # Zwrócenie ilości wykrytych liści
    return {'aspen': aspen, 'birch': birch, 'hazel': hazel, 'maple': maple, 'oak': oak}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory',
              type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path),
              required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        leaves = detect(str(img_path))
        results[img_path.name] = leaves

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
