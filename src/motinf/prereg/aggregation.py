"""Preregistered copy of aggregation helpers."""

import glob
import os

import numpy as np
import pandas as pd


def aggregate_data_prereg(in_folder: str = "data/raw/exp1/trials", out_folder: str = "data/interim/exp1") -> None:
    """Aggregate raw trial files into per-task CSVs for prereg analyses."""
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    allfolders = glob.glob(os.path.join(in_folder, "S*"))

    columns_cat = [tmpl % i for tmpl in ["seqang_%i", "seqllr_%i", "gaborimg_%i", "gaborpeakori_%i", "gaborpeakenergy_%i", "gabormeanenergynoise_%i"] for i in range(1, 13)]
    columns_est = [
        "participant",
        "date",
        "session",
        "incentive",
        "estimang",
        "estimang_noisy",
        "gaborimg_estim",
        "peakori_estim",
        "gabormeanenergynoise_estim",
        "resp_estim",
        "dev",
        "dev_noisy",
        "absdev",
        "absdev_noisy",
        "rt",
    ]

    df_all_cat = pd.DataFrame(columns=columns_cat)
    df_all_est = pd.DataFrame(columns=columns_est)

    for ifolder in allfolders:
        allfiles = glob.glob(ifolder + "/*.csv")
        df_files = pd.DataFrame(
            {
                "file_path": allfiles,
                "date": [pd.read_csv(file)["date"][0] for file in allfiles],
                "session": [pd.read_csv(file)["sessionN"][0] for file in allfiles],
            }
        )

        if len(df_files["session"].unique()) < 2:
            continue

        df_files["date"] = pd.to_datetime(df_files["date"], format="%Y-%m-%d_%Hh%M.%S.%f")
        df_files["session"] = df_files["session"].map(lambda x: 1 if x == "{{1}}" else x)
        df_files = df_files.sort_values(by=["session", "date"]).drop_duplicates(subset=["session"], keep="first")
        df_sub_est = pd.DataFrame()
        df_sub_cat = pd.DataFrame()

        for ifile in df_files["file_path"]:
            try:
                data = pd.read_csv(ifile)
                data["participant"] = data["participant"][0]
                data_main_est = data[data["trials_estim.ran"] == 1].reset_index()
                data_main_cat = data[data["trials_categ.ran"] == 1].reset_index()

                df_sess_est = pd.DataFrame()
                df_sess_est["participant"] = data_main_est["participant"].copy()
                df_sess_est["date"] = data_main_est["date"].copy()
                df_sess_est["session"] = data_main_est["sessionN"].map(lambda x: 1 if x == "{{1}}" else x).copy()
                estimang_deg = data_main_est["estimang"].to_numpy(float)
                df_sess_est["estimang"] = np.deg2rad(estimang_deg)
                df_sess_est["gaborimg_estim"] = data_main_est["gaborimg_estim"].copy()
                df_sess_est["peakori_estim"] = data_main_est["peakori_estim"].copy()
                df_sess_est["gabormeanenergynoise_estim"] = data_main_est["gabormeanenergynoise_estim"].copy()
                df_sess_est["incentive"] = data_main_est["incentive"].copy()
                df_sess_est["resp_estim"] = data_main_est["resp_estim"].copy()
                df_sess_est["rt"] = data_main_est["respEstim.stopped"].copy() - data_main_est["respEstim.started"].copy()

                df_sess_cat = pd.DataFrame()
                df_sess_cat["participant"] = data_main_cat["participant"].copy()
                df_sess_cat["date"] = data_main_cat["date"].copy()
                df_sess_cat["session"] = data_main_cat["sessionN"].map(lambda x: 1 if x == "{{1}}" else x).copy()
                df_sess_cat["incentive"] = data_main_cat["incentive"].copy()
                df_sess_cat["seqlen"] = data_main_cat["seqlen"].copy()
                df_sess_cat["sumllr"] = data_main_cat["sumllr"].copy()
                df_sess_cat["true_cat"] = 3 - 2 * data_main_cat["seqcat"].copy()
                df_sess_cat["kappa"] = data_main_cat["kappa"].copy()
                df_sess_cat["respmap"] = data_main_cat["respmap"].copy()
                df_sess_cat["resp"] = data_main_cat["resp_categ.keys"].copy()
                df_sess_cat["resp_side"] = data_main_cat["resp_categ.keys"].map({"left": -1, "right": 1}).astype(float)
                df_sess_cat["choice"] = -df_sess_cat["resp_side"] * df_sess_cat["respmap"]
                df_sess_cat["correct"] = data_main_cat["resp_categ.corr"].copy()
                df_sess_cat["rt"] = data_main_cat["resp_categ.rt"].copy()

                cols = [
                    tmpl % i
                    for tmpl in [
                        "seqang_%i",
                        "seqllr_%i",
                        "gaborimg_%i",
                        "gaborpeakori_%i",
                        "gaborpeakenergy_%i",
                        "gabormeanenergynoise_%i",
                        "gabormaxenergynoise_%i",
                        "gabornoisepeakssum_%i",
                        "gabornoisepeaksn_%i",
                    ]
                    for i in range(1, 13)
                ]
                df_sess_cat.loc[:, cols] = data_main_cat.loc[:, cols]

                seqang_noisy = (
                    df_sess_cat[[f"seqang_{i}" for i in range(1, 13)]].to_numpy(dtype=float)
                    + df_sess_cat[[f"gaborpeakori_{i}" for i in range(1, 13)]].to_numpy(dtype=float)
                )
                seqang_noisy = np.mod(seqang_noisy, np.pi)
                df_sess_cat = df_sess_cat.join(pd.DataFrame(seqang_noisy, columns=[f"seqang_noisy_{i}" for i in range(1, 13)]))
                df_sess_cat = df_sess_cat.join(
                    pd.DataFrame(
                        2 * df_sess_cat["kappa"].to_numpy(dtype=float)[:, None] * np.sin(-2 * seqang_noisy),
                        columns=[f"seqllr_noisy_{i}" for i in range(1, 13)],
                        dtype=float,
                    )
                )
                df_sess_cat["sumllr_noisy"] = np.sum([df_sess_cat[f"seqllr_noisy_{i}"] for i in range(1, 13)], 0)
                df_sess_cat["optim"] = (np.sign(df_sess_cat["choice"]) == np.sign(df_sess_cat["sumllr_noisy"])).astype(int)

                df_sess_est["estimang_noisy"] = (df_sess_est["estimang"] + df_sess_est["peakori_estim"]) % np.pi
                df_sess_est.loc[df_sess_est["resp_estim"] < 0, "resp_estim"] += np.pi
                dev = (df_sess_est["resp_estim"] - df_sess_est["estimang"]).to_numpy(float)
                dev[dev < -np.pi / 2] += np.pi
                dev[dev > np.pi / 2] -= np.pi
                df_sess_est["dev"] = dev
                dev_noisy = (df_sess_est["resp_estim"] - df_sess_est["estimang_noisy"]).to_numpy(float)
                dev_noisy[dev_noisy < -np.pi / 2] += np.pi
                dev_noisy[dev_noisy > np.pi / 2] -= np.pi
                df_sess_est["dev_noisy"] = dev_noisy
                df_sess_est["absdev"] = np.abs(dev)
                df_sess_est["absdev_noisy"] = np.abs(dev_noisy)

                df_sub_est = pd.concat([df_sub_est, df_sess_est], ignore_index=True)
                df_sub_cat = pd.concat([df_sub_cat, df_sess_cat], ignore_index=True)
            except Exception:
                print("skipped", ifile)

            df_sub_est["trialN"] = np.arange(1, len(df_sub_est) + 1)
            df_sub_cat["trialN"] = np.arange(1, len(df_sub_cat) + 1)

        df_all_est = pd.concat([df_all_est, df_sub_est], ignore_index=True)
        df_all_cat = pd.concat([df_all_cat, df_sub_cat], ignore_index=True)

    df_all_est.reset_index(drop=True).to_csv(os.path.join(out_folder, "all_trials_est.csv"), index=False)
    df_all_cat.reset_index(drop=True).to_csv(os.path.join(out_folder, "all_trials_inf.csv"), index=False)
