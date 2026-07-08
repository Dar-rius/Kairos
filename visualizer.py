import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px

class Visualizer:
    def __init__(self):
        self.colors = {0: 'red', 1: 'blue', 2: 'green'}
        self.labels = {0: 'Short', 1: 'Hold', 2: 'Buy'}
    
    #Get all buffer data that we need
    def _get_buffer_data(self, buffer):
        if buffer.slice == 0:
            return None, None, None
            
        beliefs = buffer.beliefs[:buffer.slice]
        actions = buffer.actions[:buffer.slice].astype(int)
        portfolio = buffer.portfolio[:buffer.slice].reshape(-1)
        return beliefs, actions, portfolio

    #Log a belief 3D Dscatter
    def log_belief_scatter(self, buffer):
        """
        Scatter 3D: X=P(Bull), Y=P(Bear), Z=P(Sideways)
        Color=Action, Size=Portfolio
        """
        beliefs, actions, portfolio = self._get_buffer_data(buffer)
        if beliefs is None:
            return None
            
        valid_mask = np.isfinite(portfolio) & np.all(np.isfinite(beliefs), axis=1)
        if not np.any(valid_mask):
            return None
            
        beliefs = beliefs[valid_mask]
        actions = actions[valid_mask]
        portfolio = portfolio[valid_mask]
        
        n_points = len(actions)
        if n_points == 0:
            return None
        
        if n_points > 3000:
            idx = np.random.choice(n_points, 3000, replace=False)
            beliefs, actions, portfolio = beliefs[idx], actions[idx], portfolio[idx]
            n_points = 3000
        
        # Normalize ploty size (5 to 50)
        p_min, p_max = portfolio.min(), portfolio.max()
        portfolio_clean = np.nan_to_num(portfolio, nan=p_min, posinf=p_max, neginf=p_min)
        
        if p_max > p_min:
            sizes = 5 + 45 * (portfolio_clean - p_min) / (p_max - p_min)
        else:
            sizes = np.full_like(portfolio, 20)
        
        # Mapping actions
        action_names = np.array([self.labels[a] for a in actions])
        
        # Create a new ploty figure
        fig = px.scatter_3d(
            x=beliefs[:, 0], y=beliefs[:, 1], z=beliefs[:, 2],
            color=action_names,
            size=sizes,
            color_discrete_map={
                'Short': self.colors[0],
                'Hold': self.colors[1],
                'Buy': self.colors[2]
            },
            labels={
                'x': 'P(Bull)', 'y': 'P(Bear)', 'z': 'P(Sideways)',
                'color': 'Action', 'size': 'Relative Size'
            },
            title=f'Belief Space 3D (n={n_points})',
            opacity=0.7,
            height=700, width=900
        )
        
        # Optimize layout
        fig.update_traces(marker={"line":{"width":0}})
        fig.update_layout(
            scene={
                "xaxis":{"range":[0, 1]},
                "yaxis":{"range":[0, 1]},
                "zaxis":{"range":[0, 1]},
                "aspectmode":'cube',
                "camera":{"eye":{"x":1.2, "y":1.2, "z":1.0}}
                }
        )
        
        return wandb.Html(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    
    def log_portfolio_timeline(self, buffer):
        _, actions, portfolio = self._get_buffer_data(buffer)
        if portfolio is None or len(portfolio) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        changes = np.where(actions[:-1] != actions[1:])[0] + 1
        starts = np.insert(changes, 0, 0)
        ends = np.append(changes, len(actions))
        
        # draw actions zone
        for start, end in zip(starts, ends):
            curr_action = actions[start]
            ax.axvspan(start, end - 1, alpha=0.2, color=self.colors[curr_action])
        
        # Draw portfolio
        ax.plot(portfolio, color='black', linewidth=2)
        ax.fill_between(range(len(portfolio)), portfolio, alpha=0.1, color='gray')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Portfolio ($)')
        ax.set_title('Portfolio Evolution')
        
        # Legend
        patches = [mpatches.Patch(color=self.colors[i], label=self.labels[i], alpha=0.5) 
                   for i in range(3)]
        ax.legend(handles=patches, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        img = wandb.Image(fig)
        plt.close(fig)
        
        return img
    
    def log_regime_action_dist(self, buffer):
        """actions distribution by regime"""
        beliefs, actions, _ = self._get_buffer_data(buffer)
        if actions is None or len(actions) == 0:
            return None
        
        regimes = np.argmax(beliefs, axis=1)
        regime_names = ['Bull', 'Bear', 'Sideways']
        
        data = np.zeros((3, 3))
        for r in range(3):
            mask = regimes == r
            if mask.any():
                # np.bincount compte les occurrences de 0, 1, 2 efficacement
                counts = np.bincount(actions[mask], minlength=3)[:3]
                data[r] = (counts / counts.sum()) * 100
        
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
        """Log all data about scatter 3d"""
        logs = {}
        
        scatter_3d = self.log_belief_scatter(buffer)
        if scatter_3d:
            logs['belief_space_3d'] = scatter_3d
            
        timeline = self.log_portfolio_timeline(buffer)
        if timeline:
            logs['portfolio_timeline'] = timeline
            
        regime_dist = self.log_regime_action_dist(buffer)
        if regime_dist:
            logs['regime_action_dist'] = regime_dist
            
        if logs:
            wandb.log(logs, step=global_step)
