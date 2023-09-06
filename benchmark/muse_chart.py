from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd

bar_width = 0.10


def main():
    parser = ArgumentParser()
    parser.add_argument("--graphing", required=True)

    args = parser.parse_args()

    if args.graphing == "time":
        y_axis_key = "Median"
        y_label = "Median Time (s)"
    elif args.graphing == "memory":
        y_axis_key = "Max Memory"
        y_label = "Max Memory (GB)"
    else:
        assert False, args.graphing

    df = pd.read_csv("benchmark/artifacts/all.csv")

    # round to GB
    df["Max Memory"] = df["Max Memory"].apply(lambda x: round(x / 10**9, 2))

    df["Median"] = df["Median"].apply(lambda x: round(x / 1000, 2))

    devices = df["Device"].unique()
    timesteps = [12, 20]
    resolutions = [256, 512]

    num_rows = len(devices) * len(timesteps)
    num_cols = len(resolutions)

    fig, axs = plt.subplots(num_rows, num_cols, sharey="row")

    for row_idx_1, device in enumerate(devices):
        for row_idx_2, timesteps in enumerate(timesteps):
            row_idx = row_idx_1 * len(devices) + row_idx_2

            for col_idx, resolution in enumerate(resolutions):
                legend = row_idx == 0 and col_idx == 1

                chart(
                    df=df,
                    device=device,
                    resolution=resolution,
                    plot_on=axs[row_idx, col_idx],
                    legend=legend,
                    y_axis_key=y_axis_key,
                    y_label=y_label,
                    timesteps=timesteps,
                )

    plt.subplots_adjust(hspace=0.75, wspace=0.50)

    plt.show()


def chart(df, device, resolution, plot_on, legend, y_axis_key, y_label, timesteps):
    filter = (df["Device"] == device) & (df["Resolution"] == resolution)

    if timesteps is not None:
        filter = filter & (df["Timesteps"] == timesteps)

    fdf = df[filter]

    placement = range(2)

    def inc_placement():
        nonlocal placement
        placement = [x + bar_width + 0.05 for x in placement]

    (fdf["Model Name"] == "stable_diffusion_1_5") & (fdf["Use Xformers"] == False)

    for use_xformers in [False, True]:
        filter_ = (fdf["Model Name"] == "stable_diffusion_1_5") & (fdf["Use Xformers"] == use_xformers)

        plot_one_bar(
            fdf=fdf,
            filter_=filter_,
            plot_on=plot_on,
            placement=placement,
            label=f"stable_diffusion_1_5, use_xformers: {use_xformers}",
            y_axis_key=y_axis_key,
        )

        inc_placement()

    for use_xformers, use_fused_mlp, use_fused_residual_norm in [
        [False, False, False],
        [True, False, False],
        [True, True, True],
    ]:
        filter_ = (
            (fdf["Model Name"] == "muse")
            & (fdf["Use Xformers"] == use_xformers)
            & (fdf["Use Fused MLP"] == use_fused_mlp)
            & (fdf["Use Fused Residual Norm"] == use_fused_residual_norm)
        )

        plot_one_bar(
            fdf=fdf,
            filter_=filter_,
            plot_on=plot_on,
            placement=placement,
            label=(
                f"muse, use_xformers: {use_xformers}, use_fused_mlp: {use_fused_mlp}, use_fused_residual_norm:"
                f" {use_fused_residual_norm}"
            ),
            y_axis_key=y_axis_key,
        )

        inc_placement()

    plot_on.set_xlabel("Batch Size")
    plot_on.set_ylabel(y_label)
    plot_on.set_xticks([r + bar_width for r in range(2)], [1, 8])
    plot_on.set_title(f"{device}, timesteps: {timesteps}, resolution: {resolution}")

    if legend:
        plot_on.legend(fontsize="x-small")


def plot_one_bar(fdf, filter_, plot_on, placement, label, y_axis_key):
    ffdf = fdf[filter_]

    y_axis = ffdf[y_axis_key].tolist()

    for _ in range(2 - len(y_axis)):
        y_axis.append(0)

    bars = plot_on.bar(placement, y_axis, width=bar_width, label=label)

    for bar in bars:
        yval = bar.get_height()
        plot_on.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.05,
            yval,
            ha="center",
            va="bottom",
            rotation=80,
            fontsize="small",
        )


"""
python benchmark/muse_chart.py --graphing time
python benchmark/muse_chart.py --graphing time
python benchmark/muse_chart.py --graphing memory
python benchmark/muse_chart.py --graphing memory
"""
if __name__ == "__main__":
    main()
