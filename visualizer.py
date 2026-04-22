import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import plotly.express as px
import plotly.graph_objects as go
from collections import deque, Counter
import torch

class Visualizer:
    def __init__(self):
        self.colors = {0: '#3498db', 1: '#2ecc71', 2: '#e74c3c'}
        self.labels = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
    
    def log_belief_scatter(self, buffer):
        """
        Scatter 3D: X=P(Bull), Y=P(Bear), Z=P(Sideways)
        Color=Action, Size=Portfolio
        """
        if buffer.slice == 0:
            return None
            
        # Récupération et nettoyage des données
        beliefs = buffer.beliefs[:buffer.slice].cpu().numpy()
        actions = buffer.actions[:buffer.slice].cpu().numpy()
        portfolio = buffer.portfolio[:buffer.slice].cpu().numpy()
        
        # Filtrage des NaN/Inf
        valid_mask = np.isfinite(portfolio) & np.all(np.isfinite(beliefs), axis=1)
        if not np.any(valid_mask):
            return None
            
        beliefs = beliefs[valid_mask]
        actions = actions[valid_mask]
        portfolio = portfolio[valid_mask]
        
        n_points = len(actions)
        if n_points == 0:
            return None
        
        # Sampling si trop de points (perf wandb)
        if n_points > 3000:
            idx = np.random.choice(n_points, 3000, replace=False)
            beliefs = beliefs[idx]
            actions = actions[idx]
            portfolio = portfolio[idx]
            n_points = 3000
        
        # Normalisation des tailles pour plotly (5 à 50)
        p_min, p_max = portfolio.min(), portfolio.max()
        if p_max > p_min:
            sizes = 5 + 45 * (portfolio - p_min) / (p_max - p_min)
        else:
            sizes = np.full_like(portfolio, 20)
        
        # Mapping actions
        action_names = np.array([self.labels[a] for a in actions])
        
        # Création figure Plotly
        fig = px.scatter_3d(
            x=beliefs[:, 0],
            y=beliefs[:, 1],
            z=beliefs[:, 2],
            color=action_names,
            size=sizes,
            color_discrete_map={
                'Hold': self.colors[0],
                'Buy': self.colors[1],
                'Sell': self.colors[2]
            },
            labels={
                'x': 'P(Bull)',
                'y': 'P(Bear)',
                'z': 'P(Sideways)',
                'color': 'Action',
                'size': 'Relative Size'
            },
            title=f'Belief Space 3D (n={n_points})',
            opacity=0.7,
            height=700,
            width=900
        )
        
        # Optimisation layout
        fig.update_traces(marker=dict(line=dict(width=0)))
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                zaxis=dict(range=[0, 1]),
                aspectmode='cube',
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.0))
            )
        )
        
        return wandb.Html(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    
    
    
    def log_portfolio_timeline(self, buffer):
        """Timeline portfolio avec zones d'action"""
        portfolio = buffer.portfolio[:buffer.slice].cpu().numpy()
        actions = buffer.actions[:buffer.slice].cpu().numpy()
        
        if len(portfolio) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Zones d'actions consécutives
        start_idx = 0
        curr_action = int(actions[0])
        
        for i in range(1, len(actions)):
            if int(actions[i]) != curr_action or i == len(actions) - 1:
                ax.axvspan(start_idx, i, alpha=0.2, color=self.colors[curr_action])
                start_idx = i
                curr_action = int(actions[i])
        
        ax.plot(portfolio, color='black', linewidth=2)
        ax.fill_between(range(len(portfolio)), portfolio, alpha=0.1, color='gray')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Portfolio ($)')
        ax.set_title('Portfolio Evolution')
        
        patches = [mpatches.Patch(color=self.colors[i], label=self.labels[i], alpha=0.5) 
                  for i in range(3)]
        ax.legend(handles=patches, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        img = wandb.Image(fig)
        plt.close(fig)
        return img
    
    def log_regime_action_dist(self, buffer):
        """Distribution des actions par régime"""
        beliefs = buffer.beliefs[:buffer.slice].cpu().numpy()
        actions = buffer.actions[:buffer.slice].cpu().numpy()
        
        if len(actions) == 0:
            return None
        
        regimes = np.argmax(beliefs, axis=1)
        regime_names = ['Bull', 'Bear', 'Sideways']
        
        data = np.zeros((3, 3))
        for r in range(3):
            mask = regimes == r
            if mask.sum() > 0:
                for a in range(3):
                    data[r, a] = (actions[mask] == a).mean() * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(3)
        width = 0.25
        
        for i in range(3):
            ax.bar(x + i*width, data[:, i], width, 
                   label=self.labels[i], color=self.colors[i], alpha=0.8)
        
        ax.set_ylabel('Frequency (%)')
        ax.set_title('Action Distribution by Regime')
        ax.set_xticks(x + width)
        ax.set_xticklabels(regime_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        img = wandb.Image(fig)
        plt.close(fig)
        return img
    
    def log_all(self, buffer, global_step=None):
        """Logue tous les graphes incluant le scatter 3D"""
        logs = {}
        
        # Scatter 3D (Plotly interactif)
        scatter_3d = self.log_belief_scatter(buffer)
        if scatter_3d:
            logs['belief_space_3d'] = scatter_3d
        
        # Timeline
        timeline = self.log_portfolio_timeline(buffer)
        if timeline:
            logs['portfolio_timeline'] = timeline
        
        # Régime/Action
        regime_dist = self.log_regime_action_dist(buffer)
        if regime_dist:
            logs['regime_action_dist'] = regime_dist
        
        if logs:
            wandb.log(logs, step=global_step)
