import matplotlib.pyplot as plt

def plot_pls_scores(pls_model, y, component_x=1, component_y=2,
                    group_names=("Low", "High")):
    """
    Plot any two PLS latent components.

    Parameters
    ----------
    component_x : int
        Component to plot on the x-axis (1-based).

    component_y : int
        Component to plot on the y-axis (1-based).
    """

    scores = pls_model.x_scores_

    if scores.shape[1] < max(component_x, component_y):
        print(f"Model only has {scores.shape[1]} component(s).")
        return

    x = component_x - 1
    y_idx = component_y - 1

    plt.figure(figsize=(6,6))

    plt.scatter(
        scores[y == 0, x],
        scores[y == 0, y_idx],
        alpha=0.7,
        label=group_names[0]
    )

    plt.scatter(
        scores[y == 1, x],
        scores[y == 1, y_idx],
        alpha=0.7,
        label=group_names[1]
    )

    plt.xlabel(f"PLS Component {component_x}")
    plt.ylabel(f"PLS Component {component_y}")
    plt.title("PLS-DA Score Plot")
    plt.legend()
    plt.tight_layout()
    plt.show()