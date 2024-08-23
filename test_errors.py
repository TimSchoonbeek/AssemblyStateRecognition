import numpy as np
import argparse
import json
from pathlib import Path
import pandas as pd
import torch

from utils import DEVICE, compute_embeddings, load_run_model, \
    get_error_embeddings, get_rng, binary_verification_for_errors
from datasets import RealContrastiveDatasetWithInters, ErrorDataset, get_transform, get_image
from torch.utils.data import DataLoader


def save_embeddings(test_name, model, data_path, img_size, save_dir):
    print(f"Loading embeddings from {data_path}")
    if test_name == "sim2real":
        dataset = RealContrastiveDatasetWithInters(data_path.parent.parent / "industreal_cont" / "train", w=img_size[0], h=img_size[1], skip_factor=50, only_clean=True)
        emb_path = save_dir / "embeddings.csv"
        lab_path = save_dir / "labels.csv"
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=True)
        print(f"Successfully loaded dataset and dataloader. Running inference...")
        embeddings, labels = compute_embeddings(data_loader, model)

    elif test_name == "errors":
        dataset = ErrorDataset(data_path / "assembly_states_errors", get_transform(train=False, synth=False),
                               w=img_size[0], h=img_size[1])
        data_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=8, drop_last=True)
        print(f"Successfully loaded dataset and dataloader. Running inference...")

        embeddings, intended_states, error_cats = get_error_embeddings(data_loader, model)
        labels = np.stack((intended_states, error_cats))

        emb_path = save_dir / "error_embeddings.csv"
        lab_path = save_dir / "error_labels.csv"
    else:
        raise NotImplementedError(f"Only implemented generalization test, not {test_name}")

    print(f"Saving {labels.shape[0]} embeddings and labels...")
    df = pd.DataFrame(embeddings)
    df.to_csv(emb_path, header=False, index=False, float_format='%.5f')
    df = pd.DataFrame(labels)
    df.to_csv(lab_path, header=False, index=False, float_format='%.5f')
    print(f"Saved embeddings and labels to {save_dir}")
    return embeddings, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=str, help='Path to the run directory, e.g. ./runs/run_name')
    parser.add_argument("--checkpoint", type=str, default=None, help='Name of the checkpoint to be tested')
    parser.add_argument("--dist", type=str, default="cos", help='l2 or cos (cosine) distance. default = cos')
    parser.add_argument("--data_path", type=str, default="./data", help='Location of the data dir')
    parser.add_argument("--img_w", type=int, default=224,
                        help='width and height of the image inputs to the model.')
    parser.add_argument("--img_h", type=int, default=224,
                        help='width and height of the image inputs to the model.')
    test_args = parser.parse_args()

    rng = get_rng()

    run_path = Path(test_args.run_path)
    data_path = Path(test_args.data_path)
    if not run_path.exists():
        raise ValueError(f"The run {test_args.run_name} you are trying to test, does not exist!")

    if test_args.checkpoint is None:
        checkpoints = ["best.pth"]
    else:
        checkpoints = [test_args.checkpoint]

    for checkpoint in checkpoints:
        print(f"Evaluating checkpoint {checkpoint}")

        model_weight_path = run_path / "checkpoints" / checkpoint
        save_dir = run_path / checkpoint
        save_dir.mkdir(parents=True, exist_ok=True)

        # anchor embeddings are used only for sim2real and error tests.
        path_to_anchor_embedding = save_dir / "anchor_embeddings.csv"
        path_to_anchor_labels = save_dir / "anchor_labels.csv"

        path_to_embedding = save_dir / "error_embeddings.csv"
        path_to_labels = save_dir / "error_labels.csv"

        # create anchor embeddings if they do not yet exist
        if not path_to_anchor_embedding.exists():
            print(f"Creating anchor embeddings...")
            model = load_run_model(run_path, model_weight_path)
            model.use_projection_head(False)
            model.eval()
            model.to(DEVICE)

            anchor_path = data_path / "anchors"
            print(f"Loading anchors from: {anchor_path}")
            with open(anchor_path / "labels.json") as file:
                annotations = json.load(file)
            annotations['labels'] = np.array(annotations['labels'])

            transform = get_transform(train=False, synth=True)
            imgs = torch.zeros((len(annotations["images"]), 3, test_args.img_h, test_args.img_w))
            anchor_labels = np.zeros(len(annotations["images"]), dtype=int)
            for i, anchor_name in enumerate(annotations["images"]):
                target = annotations['labels'][i]
                img = get_image(anchor_path / "images" / anchor_name, size=(test_args.img_w, test_args.img_h))
                img = transform(img)
                imgs[i, :, :, :] = img
                anchor_labels[i] = target

            anchor_embeddings = model(imgs.to(DEVICE))
            anchor_embeddings = anchor_embeddings.detach().cpu().numpy()

            df = pd.DataFrame(anchor_embeddings)
            df.to_csv(path_to_anchor_embedding, header=False, index=False, float_format='%.5f')

            df = pd.DataFrame(anchor_labels)
            df.to_csv(path_to_anchor_labels, header=False, index=False, float_format='%.5f')
            print(f"Saved anchor embeddings to {save_dir}/anchor_embeddings.csv")
        else:
            print(f"Anchor embeddings are already created, so not running inference again. Loading embedddings...")
            anchor_embeddings = np.genfromtxt(path_to_anchor_embedding, delimiter=',')
            anchor_labels = np.genfromtxt(path_to_anchor_labels, delimiter=',')
            print(f"Loaded anchor embeddings and labels")

            model = None

        if not path_to_embedding.exists():
            if model is None:
                model = load_run_model(run_path, model_weight_path)
                model.use_projection_head(False)
                model.eval()
                model.to(DEVICE)

            save_embeddings("sim2real", model, data_path, (test_args.img_w, test_args.img_h), save_dir)
            embeddings, labels = save_embeddings("errors", model, data_path, (test_args.img_w, test_args.img_h), save_dir)
        else:
            print(f"Embeddings are already created, so not running inference again. Loading embedddings...")
            embeddings = np.genfromtxt(path_to_embedding, delimiter=',')
            labels = np.genfromtxt(path_to_labels, delimiter=',')
            print(f"Loaded embeddings and labels")

        print(f"Running test - errors")
        intended_states = labels[0, :]
        error_categories = labels[1, :]

        correct_embeddings = np.genfromtxt(save_dir / "embeddings.csv", delimiter=',')
        correct_labels = np.genfromtxt(save_dir / "labels.csv", delimiter=',')

        mAP, ROC, scores = binary_verification_for_errors(rng, embeddings, intended_states, error_categories,
                                                          correct_embeddings, correct_labels, anchor_embeddings,
                                                          anchor_labels, test_args.dist)
        print(f"mAP for binary state verification: {mAP:.3f}")
        print(f"auROC for binary state verification: {ROC:.3f}")
        rocs = []
        aps = []
        for d in scores:
            print(
                f"{scores[d]['name']}:\t mAP: {scores[d]['mAP'] * 100:.2f} \t auROC: {scores[d]['ROC'] * 100:.2f} \t Samples: {int(scores[d]['n'] / 2)}")
            rocs.append(scores[d]['ROC'])
            aps.append(scores[d]['mAP'])
        print(f"Macro average AP: {sum(aps) / len(aps) * 100:.2f}")
        print(f"Macro average auROC: {sum(rocs) / len(rocs) * 100:.2f}")
        log_str = f"{checkpoint} \t errors \t mAP micro: {mAP:.3f} \t mAP macro: {sum(aps) / len(aps):.3f} \t auROC: {ROC:.3f} \n"

        print(log_str)
        file = open(run_path / 'test_log.txt', 'a')
        file.write(log_str)
        file.close()


