import matplotlib.pyplot as plt
import numpy as np
import os

def plot_survival_probabilities(survival_scores_all, censor_all, survtime_all, case_id_all, num_cases=10, output_dir='plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_cases = min(num_cases, len(case_id_all))

    for i in range(num_cases):
        fig, ax = plt.subplots(figsize=(10, 5))

        survival_scores = survival_scores_all[i]
        survival_scores = np.append(survival_scores, survival_scores[-1])  # Extend line to last period
        survival_time = survtime_all[i]
        censor = censor_all[i]
        case_id = case_id_all[i]

        # Time bins for 20 periods, each representing half a year
        time_bins = np.arange(len(survival_scores) + 1) * 0.5  # Multiply by 0.5 to represent half years

        # Plot predicted survival probabilities
        ax.step(time_bins[:-1], survival_scores, where='post', label='Predicted Survival Probabilities')

        # Add vertical line for true survival time, properly adjusted to the 6-month period scale
        survival_years = survival_time / 365  # Convert days to years
        ax.axvline(x=survival_years, color='green' if censor else 'red', linestyle='--', 
                   label='True Survival Time (Last Visit - Censored)' if censor else 'True Survival Time (Death)')

        ax.set_title(f"Case ID: {case_id}")
        ax.set_xlabel('Years')
        ax.set_ylabel('Predicted Survival Probability')
        ax.legend()

        # Adjusting x-axis to show labels in years but represent them at correct intervals
        ax.set_xticks(np.arange(0, 10.5, 1))  # Set ticks every 1 year
        ax.set_xticklabels([f"{int(x)}" for x in np.arange(0, 10.5, 1)])  # Label ticks as full years

        # Save the figure
        fig_path = os.path.join(output_dir, f"Case_{case_id}.png")
        fig.savefig(fig_path)
        plt.close(fig)  # Close the figure to free up memory
