# modify from: https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

import pandas as pd
from pathlib import Path

from torchvision.datasets.utils import download_and_extract_archive


class TMHINTQI:
    """`The TMHINTQI Data Set <https://github.com/yuwchen/InQSS>`_.
    Args:
        root (string): Root directory of the dataset.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _DATA_URL = "https://drive.google.com/file/d/1TMDiz6dnS76hxyeAcCQxeSqqEOH4UDN0"
    _DATA_MD5 = "21a6803900516734bc5fed2814eae296"

    def __init__(
        self,
        root: str,
        download: bool = False,
    ) -> None:
        self._base_folder = Path(root) / "tmhintqi"
        self._meta_folder = self._base_folder / "meta"
        self._audio_folder = self._base_folder / "TMHINTQI"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._process_and_save_meta()

    def _check_exists(self) -> bool:
        return self._audio_folder.exists() and self._audio_folder.is_dir()

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._DATA_URL, download_root=self._base_folder, filename="TMHINTQI.zip", md5=self._DATA_MD5)

    def _process_and_save_meta(self) -> None:
        data_df = pd.read_csv(self._base_folder / "raw_data.csv")

        data_df = data_df[["file_name", "quality_score"]].rename(columns={"quality_score": "rating"})
        data_df = data_df.dropna()
        data_df.rating = data_df.rating.astype(int)
        data_df.file_name = data_df.file_name.apply(lambda x: x + ".wav")
        rating_stats = data_df.groupby('file_name')['rating'].agg(['mean', 'std', 'sem', 'count'])
        rating_dist = data_df.groupby('file_name')['rating'].value_counts().unstack(fill_value=0)
        rating_dist = rating_dist.rename(columns={r: f"rating_{r}" for r in rating_dist.columns})
        meta_df = pd.merge(rating_stats, rating_dist, left_index=True, right_index=True)
        meta_df.reset_index(inplace=True)
        meta_df.columns.name = None

        train_file_names = [p.name for p in (self._audio_folder / "train").glob("*.wav")]
        train_df = meta_df[meta_df.file_name.isin(train_file_names)].copy()
        train_df.file_name = train_df.file_name.apply(lambda x: Path("train") / x)
        test_file_names = [p.name for p in (self._audio_folder / "test").glob("*.wav")]
        test_df = meta_df[meta_df.file_name.isin(test_file_names)].copy()
        test_df.file_name = test_df.file_name.apply(lambda x: Path("test") / x)

        self._meta_folder.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(self._meta_folder / "train.csv", index=False)
        test_df.to_csv(self._meta_folder / "test.csv", index=False)


if __name__ == "__main__":
    TMHINTQI("./audio", download=True)
