from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("benchmark/artifacts/all.csv")

# round to GB
df["Max Memory"] = df["Max Memory"].apply(lambda x: round(x / 10**9, 2))

df["Median"] = df["Median"].apply(lambda x: round(x/1000, 2))
df["Mean"] = df["Mean"].apply(lambda x: round(x/1000, 2))

bar_width = 0.10

model_names = [
    "runwayml/stable-diffusion-v1-5",
    "williamberman/muse_research_run",
]


def chart(device, component, resolution, plot_on, legend, y_axis_key, y_label, timesteps):
    filter = (df["Device"] == device) & (df["Component"] == component) & (df["Resolution"] == resolution)

    if timesteps is not None:
        filter = filter & (df["Timesteps"] == timesteps)

    fdf = df[filter]

    placement = range(2)

    def inc_placement():
        nonlocal placement
        placement = [x + bar_width + 0.05 for x in placement]

    for model_name in model_names:
        filter_ = fdf["Model Name"] == model_name

        ffdf = fdf[filter_]

        y_axis = ffdf[y_axis_key].tolist()

        for _ in range(2 - len(y_axis)):
            y_axis.append(0)

        bars = plot_on.bar(placement, y_axis, width=bar_width, label=f"{model_name}")

        for bar in bars:
            yval = bar.get_height()
            plot_on.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 0.05,
                yval,
                ha="center",
                va="bottom",
                rotation=80,
                fontsize="small"
            )

        inc_placement()

    plot_on.set_xlabel("Batch Size")
    plot_on.set_ylabel(y_label)
    plot_on.set_xticks([r + bar_width for r in range(2)], [1, 8])
    plot_on.set_title(f"{device}, timesteps: {timesteps}, resolution: {resolution}")

    if legend:
        plot_on.legend(fontsize="x-small")


"""
python muse_chart.py --component full --graphing time --timesteps 12
python muse_chart.py --component full --graphing time --timesteps 20
python muse_chart.py --component full --graphing memory --timesteps 12
python muse_chart.py --component full --graphing memory --timesteps 20

python muse_chart.py --component backbone --graphing time
python muse_chart.py --component backbone --graphing memory

python muse_chart.py --component vae --graphing time
python muse_chart.py --component vae --graphing memory
"""
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--component", required=True)
    parser.add_argument("--graphing", required=True)
    parser.add_argument("--timesteps", required=False, default=None)

    args = parser.parse_args()

    assert args.component in ["full", "backbone", "vae"]

    if args.graphing == "time":
        y_axis_key = "Median"
        y_label = "Median Time (s)"
    elif args.graphing == "memory":
        y_axis_key = "Max Memory"
        y_label = "Max Memory (GB)"
    else:
        assert False, args.graphing

    fig, axs = plt.subplots(4, 2, sharey="row")

    for row_idx_1, device in enumerate(["a100", "4090"]):
        for row_idx_2, timesteps in enumerate([12, 20]):

            row_idx = row_idx_1 * 2 + row_idx_2

            for col_idx, resolution in enumerate([256, 512]):
                legend = row_idx == 0 and col_idx == 1

                chart(
                    device,
                    args.component,
                    resolution,
                    axs[row_idx, col_idx],
                    legend,
                    y_axis_key,
                    y_label,
                    timesteps,
                )

    plt.subplots_adjust(hspace=0.75, wspace=0.50)

    plt.show()
