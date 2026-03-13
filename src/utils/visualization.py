"""
Visualization/Plotting Module.
============================================================================
Paper Reference: Figures 3-8, Pages 9-13
============================================================================
Reproduces the following figures from the paper:
- Figs 3.a, 4.a, 5.a, 6.a: Accuracy vs Training Round (Pages 9-12)
- Figs 3.b, 4.b, 5.b, 6.b: Loss vs Training Round (Pages 9-12)
- Figs 3.c, 4.c, 5.c, 6.c: Attention Value vs Training Round (Pages 9-12)
- Fig 7: ROC curves (Page 13)
- Fig 8: Average accuracy vs Number of agents (Page 13)

"For all the experiments, we plotted the accuracy of the model, loss obtained
 and the attention values for each of the agents after every round of training."
 (Section 6.1, Page 8)
============================================================================
"""

import matplotlib
import os
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or not os.environ.get('DISPLAY', ''):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_agent_accuracies(round_accuracies, num_agents, title_suffix="",
                          save_path=None):
    """
    Plot accuracy vs training round for each agent.

    Paper Reference: Figs 3.a, 4.a, 5.a, 6.a (Pages 9-12)
    "Figs 3.a and 4.a corresponds to the graphs plotted against the accuracy
     and training round number for each of the eight agents." (Page 8)

    Args:
        round_accuracies: Dict {agent_id: [acc_round1, acc_round2, ...]}
        num_agents: Number of agents
        title_suffix: Additional title text
        save_path: Optional path to save the figure
    """
    ncols = max(1, (num_agents + 1) // 2)
    nrows = 1 if num_agents <= 1 else 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8), squeeze=False)
    axes = axes.flatten()

    for i in range(num_agents):
        ax = axes[i]
        rounds = range(1, len(round_accuracies[i]) + 1)
        ax.plot(rounds, round_accuracies[i], 'b-', linewidth=1.5)
        ax.set_title(f'Agent {i}', fontsize=10)
        ax.set_xlabel('Round')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

    for j in range(num_agents, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'Accuracy vs Training Round {title_suffix}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved accuracy plot to {save_path}")
    plt.show()
    plt.close(fig)


def plot_agent_losses(round_losses, num_agents, title_suffix="",
                      save_path=None):
    """
    Plot loss vs training round for each agent.

    Paper Reference: Figs 3.b, 4.b, 5.b, 6.b (Pages 9-12)
    "Figs 3.b and 4.c show the plots for loss vs. round number." (Page 8)

    Args:
        round_losses: Dict {agent_id: [loss_round1, loss_round2, ...]}
        num_agents: Number of agents
        title_suffix: Additional title text
        save_path: Optional path to save the figure
    """
    ncols = max(1, (num_agents + 1) // 2)
    nrows = 1 if num_agents <= 1 else 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8), squeeze=False)
    axes = axes.flatten()

    for i in range(num_agents):
        ax = axes[i]
        rounds = range(1, len(round_losses[i]) + 1)
        ax.plot(rounds, round_losses[i], 'r-', linewidth=1.5)
        ax.set_title(f'Agent {i}', fontsize=10)
        ax.set_xlabel('Round')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)

    for j in range(num_agents, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'Loss vs Training Round {title_suffix}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved loss plot to {save_path}")
    plt.show()
    plt.close(fig)


def plot_agent_attention_values(round_attention_values, num_agents,
                                 title_suffix="", save_path=None):
    """
    Plot attention value vs training round for each agent.

    Paper Reference: Figs 3.c, 4.c, 5.c, 6.c (Pages 9-12)
    "Figs 3.c and 4.c shows the plots against attention value of each
     of the agent vs. round number of the training process." (Page 8)

    "By observing the plots we can infer that ... the attention values of
     the agents are also getting dropped with increase in the training round
     number. This is expected because of the fact that the attention value
     of an agent is dynamically determined by the model accuracy along with
     the number of training data samples in the current round." (Page 8)

    Args:
        round_attention_values: Dict {agent_id: [attn_round1, attn_round2, ...]}
        num_agents: Number of agents
        title_suffix: Additional title text
        save_path: Optional path to save the figure
    """
    ncols = max(1, (num_agents + 1) // 2)
    nrows = 1 if num_agents <= 1 else 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8), squeeze=False)
    axes = axes.flatten()

    for i in range(num_agents):
        ax = axes[i]
        rounds = range(1, len(round_attention_values[i]) + 1)
        ax.plot(rounds, round_attention_values[i], 'g-', linewidth=1.5)
        ax.set_title(f'Agent {i}', fontsize=10)
        ax.set_xlabel('Round')
        ax.set_ylabel('Attention Value')
        ax.grid(True, alpha=0.3)

    for j in range(num_agents, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'Attention Value vs Training Round {title_suffix}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved attention plot to {save_path}")
    plt.show()
    plt.close(fig)


def plot_roc_curve(fpr_arr, tpr_arr, auc_value, title="ROC Curve",
                   save_path=None):
    """
    Plot ROC curve.

    Paper Reference: Figure 7, Page 13 (Section 6, Page 8)
    "Receiver Operating Characteristic (ROC) Curve: It is a curve that is
     drawn with false positive rate (FPR) on x-axis and true positive rate
     (TPR) on the y-axis." (Page 8)

    Args:
        fpr_arr: Array of false positive rates
        tpr_arr: Array of true positive rates
        auc_value: Area under the ROC curve
        title: Plot title
        save_path: Optional save path
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_arr, tpr_arr, 'b-', linewidth=2,
             label=f'ROC curve (AUC = {auc_value:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved ROC curve to {save_path}")
    plt.show()
    plt.close()


def plot_accuracy_vs_num_agents(agent_counts, avg_accuracies, title="",
                                 save_path=None):
    """
    Plot average accuracy vs number of agents.

    Paper Reference: Figure 8, Page 13 (Section 6.3, Page 11)
    "we performed an experiment in which we simulated the system multiple
     times with varying number of agents in an increasing fashion. During
     each simulation, we randomly divided the available data into approximately
     equal sized chunks to be distributed among the agents, and subsequently
     computed the overall average accuracy of the system." (Page 11)

    "By observing the plots, we can interpret that the average accuracy of
     the system remains constant, when the number of agents increases"
     (Page 11)

    Args:
        agent_counts: List of number-of-agents values [2, 4, 6, 8, ...]
        avg_accuracies: List of corresponding average accuracies
        title: Plot title
        save_path: Optional save path
    """
    plt.figure(figsize=(8, 6))
    plt.plot(agent_counts, avg_accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Agents')
    plt.ylabel('Average Accuracy')
    plt.title(f'Average Accuracy vs Number of Agents {title}')
    plt.grid(True, alpha=0.3)
    plt.ylim([0.8, 1.05])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved scalability plot to {save_path}")
    plt.show()
    plt.close()


def plot_all_training_curves(history, num_agents, title_suffix="",
                              save_dir=None):
    """
    Plot all training curves (accuracy, loss, attention) in one call.

    Paper Reference: Figures 3-6, Pages 9-12
    Each figure has 3 sub-figures: accuracy, loss, and attention value
    plotted against training round number for each agent.

    Args:
        history: Training history dict from orchestrator
        num_agents: Number of agents
        title_suffix: Additional title text
        save_dir: Optional directory to save figures
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    acc_path = os.path.join(save_dir, 'accuracy_vs_round.png') if save_dir else None
    loss_path = os.path.join(save_dir, 'loss_vs_round.png') if save_dir else None
    attn_path = os.path.join(save_dir, 'attention_vs_round.png') if save_dir else None

    plot_agent_accuracies(
        history['round_accuracies'], num_agents,
        title_suffix=title_suffix, save_path=acc_path
    )

    plot_agent_losses(
        history['round_losses'], num_agents,
        title_suffix=title_suffix, save_path=loss_path
    )

    plot_agent_attention_values(
        history['round_attention_values'], num_agents,
        title_suffix=title_suffix, save_path=attn_path
    )
