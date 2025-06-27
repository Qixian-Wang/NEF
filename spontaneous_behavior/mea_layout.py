import random
import matplotlib.pyplot as plt

def point_in_polygon(x, y, polygon):
    inside = False
    n = len(polygon)
    for i in range(n):
        pointa_x, pointa_y = polygon[i]
        pointb_x, pointb_y = polygon[(i + 1) % n]
        if ((pointa_y > y) != (pointb_y > y)):
            xinters = (pointb_x - pointa_x) * (y - pointa_y) / (pointb_y - pointa_y) + pointa_x
            if x < xinters:
                inside = not inside
    return inside

def plot_topology(polygon, electrode_positions, neuron_positions):
    plt.figure(figsize=(8, 10))
    # Polygon
    xs, ys = zip(*(polygon+[polygon[0]]))
    plt.plot(xs, ys, '-k', linewidth=2)

    # Electrode
    for idx, (x, y) in enumerate(electrode_positions):
        plt.scatter(x, y, marker='o', s=100, color='blue')
        plt.text(
            x, y,
            str(idx),
            fontsize=8,
            ha='center', va='center',
            color='white',
            bbox=dict(
                boxstyle='round,pad=0.2',
                fc='blue',
                alpha=0.7
            )
        )
    # Neuron
    nx, ny = zip(*neuron_positions)
    plt.scatter(nx, ny, marker='o', s=5, alpha=0.6, color='yellow')

    plt.gca().set_aspect('equal', 'box')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Topology Display')
    plt.grid(True)
    plt.show()

def generate_layout(
    height=1500,
    electrode_b=32,
    electrode_t=16,
    electrode_per_row_b=4,
    electrode_per_row_t=4,
    num_neuron=3000,
    spacing=100,
    plot_result=False,
):
    height = height
    length_b = (electrode_b // electrode_per_row_b + 1) * spacing
    length_t = (electrode_t // electrode_per_row_t + 1) * spacing

    x_offset_top = (length_b - length_t) / 2

    electrode_positions = []
    neuron_positions = []

    blocks = [
        (electrode_b // electrode_per_row_b, electrode_per_row_b, 0, spacing/2),
        (electrode_t // electrode_per_row_t, electrode_per_row_t,
         x_offset_top, height - spacing/2 - (electrode_per_row_t - 1)*spacing)
    ]

    for n_cols, n_rows, x_off, margin in blocks:
        for col in range(n_cols):
            for row in range(n_rows):
                x = spacing * col + x_off
                y = margin + spacing * row
                electrode_positions.append((x, y))

    polygon = [
        (-spacing, 0),
        (length_b - spacing, 0),
        (length_t - spacing + x_offset_top, height),
        (-spacing + x_offset_top, height)
    ]

    min_x = -spacing
    max_x = length_b - spacing
    min_y = 0
    max_y = height
    for _ in range(num_neuron):
        while True:
            x_rand = random.uniform(min_x, max_x)
            y_rand = random.uniform(min_y, max_y)
            if point_in_polygon(x_rand, y_rand, polygon):
                neuron_positions.append((x_rand, y_rand))
                break

    if plot_result:
        plot_topology(polygon, electrode_positions, neuron_positions)

    return electrode_positions, neuron_positions

if __name__ == "__main__":
    coords = generate_layout(
        height=1500,
        electrode_t=16,
        electrode_b=32,
        electrode_per_row_b=4,
        electrode_per_row_t=4,
        num_neuron=3000,
        spacing=100,
        plot_result=True
    )
    print(coords)