"""Download the Zenodo 14983362 files and verify them against published MD5s.

The MD5 values below were copied from the file listing on the Zenodo record
page (pasted by the user on 2026-10-13). A file whose MD5 does not match is
reported and kept under a .bad suffix, never silently accepted.

Usage (from the project root):
    py scripts/download_zenodo_state_daily.py
Already-downloaded files with a matching MD5 are skipped.
"""
import hashlib
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "zenodo_state_daily"
URL = "https://zenodo.org/records/14983362/files/{}?download=1"

MD5 = {
    "Andaman_and_Nicobar.csv": "adf6074fd5a72e4317867cd73992f36c",
    "Andhra_Pradesh.csv": "26a68b42856ded54310919bf420e0871",
    "Arunachal_Pradesh.csv": "b35ab34ca0bbff00f5edc0d69c5c3d32",
    "Assam.csv": "37a691b880d0a990a1fd2dde44ee7ac1",
    "Bihar.csv": "6efb2668a92797e31fe06ca6ede58795",
    "Chandigarh.csv": "869ce1c99821fa6a188ee018461ec3f5",
    "Chhattisgarh.csv": "907bfe690d36c500087b70172901a384",
    "corrected_daily_energy_met_MU.csv": "172540c2c13282cc3dd4c9679b537aa0",
    "corrected_peak_met_MW.csv": "26f1531a7e045f78fb7a7c7f6e008a9d",
    "Dadra_and_Nagar_Haveli.csv": "1bbd822ff80fb39b311dfe0d39c7be1d",
    "daily_energy_met_MU.csv": "c6d677992d4cf6735d09b71823140468",
    "Daman_and_Diu.csv": "1bbd822ff80fb39b311dfe0d39c7be1d",
    "Goa.csv": "d00464bbf365e2eff4ef34fdd83523df",
    "Gujarat.csv": "9f05ad81914a43a353c85666189f2433",
    "Haryana.csv": "64b030ad73415512a213917f4c1d46ca",
    "Himachal_Pradesh.csv": "207d3d4cf3b4387cbb84c82b87b7fd8a",
    "Jammu_and_Kashmir.csv": "a56967c5b3fdc1e42f35355433e31733",
    "Jharkhand.csv": "647b6972ce52a20ab0c4eb09e3b69cae",
    "Karnataka.csv": "6f42f9c502cb503e60d94369fe939aee",
    "Kerala.csv": "12ac3ecf4bdcc3dfe5bdc0ca2ce13d03",
    "Lakshadweep.csv": "7a3e098cf1a876c21917882f6791b2a8",
    "Madhya_Pradesh.csv": "6e0994a9633bcbf542fd4cc3d8ed444a",
    "Maharashtra.csv": "cc7ddf4d2cbf2d6ed8ef24a0f0a1caf4",
    "Manipur.csv": "5b2c590124ac23a50cb8010578527615",
    "Meghalaya.csv": "57ebc831d1dd8f43a78b948e720953ee",
    "Mizoram.csv": "84362ce3ee29f1a9daa9d8f327b6c593",
    "Nagaland.csv": "3df89750e37a9cfbb9f541d5f04f7a12",
    "NCT_of_Delhi.csv": "14c358e9c5938c134497d37d51c77d59",
    "Odisha.csv": "6c375ea620c23c365c3a0de37f88b493",
    "peak_met_MW.csv": "40e83946b1b98f7dc7098071d3e29c78",
    "peak_unmet_MW.csv": "4dcbea56b60be4d2cb4232710c163896",
    "Puducherry.csv": "c890473ec0cb24c31094c4aff91ee851",
    "Punjab.csv": "22401296d70aeff6bcfeb4dce25cae42",
    "Rajasthan.csv": "9fcf3141bfb70a1868aaaee4c8369211",
    "Sikkim.csv": "db95e68d2b36c6b65760b00ca650a8d4",
    "Tamil_Nadu.csv": "6e8c6d1b2788422c43dd72b97ef134d7",
    "Telangana.csv": "e89000583757bf1a3cfd42a895878ebd",
    "Tripura.csv": "cba814ffc2364f30971bc93fb662104f",
    "Uttar_Pradesh.csv": "0e80d8cd1ae27a3360f02aaf4a3b0571",
    "Uttarakhand.csv": "d97ba64a0f3b4373a0d0f1e04f2ccdb4",
    "weighted_daily_mean_utci_india.csv": "5623b55e693f4bfedb70966956c8fe04",
    "West_Bengal.csv": "351e714a85dad8ded32a4b94355c1c83",
}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bad = []
    for name, want in MD5.items():
        dest = OUT / name
        if dest.exists() and md5_of(dest) == want:
            print(f"ok (cached)  {name}")
            continue
        r = requests.get(URL.format(name), timeout=300)
        r.raise_for_status()
        dest.write_bytes(r.content)
        got = md5_of(dest)
        if got == want:
            print(f"ok           {name}  {len(r.content)/1e6:.1f} MB")
        else:
            dest.rename(dest.with_suffix(".csv.bad"))
            bad.append(name)
            print(f"MD5 MISMATCH {name}: expected {want}, got {got}")
    print(f"\n{len(MD5) - len(bad)}/{len(MD5)} files verified")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
