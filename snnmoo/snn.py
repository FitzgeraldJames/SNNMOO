import numpy as np
import pandas as pd

import chart_studio.plotly as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot

from typing import List, Tuple, Iterator, Callable, Dict, Any
from dataclasses import dataclass


@dataclass
class SNNFirings:
    '''Stores the results from an instantiation of the SNN and the parameters used'''
    ge:         float
    gi:         float
    Ne:         int  
    Ni:         int  
    time:       int  
    sparsity:   float
    thalmic_ex: float
    thalmic_in: float
    firings:    List[List[int]]
    
    def _firings_to_spikes_at_time(self) -> pd.DataFrame:
        '''spikes at time statistics'''
        firings = self.firings 

        spikes_at_time = []
        for t, fired in enumerate(firings):
            num_fired  = len(fired)
            ex_firings = sum(1 for neuron in fired if neuron < self.Ne)
            in_firings = num_fired - ex_firings

            spikes_at_time.append({
                'time': t, 
                'num_fired': num_fired,
                'excitatory': ex_firings,
                'inhibitory': in_firings
            })

        return pd.DataFrame(spikes_at_time)

    def _firings_time_bin(self, interval: int) -> pd.DataFrame:
        '''Convert spikes at time to time bin of firings'''
        spikes_at_time = self._firings_to_spikes_at_time()
        spikes_at_time['bin'] = spikes_at_time['time'] // interval

        bins = []
        for time, time_bin in spikes_at_time.groupby('bin'):
            time = min(time_bin['time'])

            bins.append({
                'time': time, 
                'num_fired': time_bin['num_fired'].mean(),
                'excitatory': time_bin['excitatory'].mean(),
                'inhibitory': time_bin['inhibitory'].mean()
            })

        return pd.DataFrame(bins)
            

    def _flatten_firings(self) -> List[Tuple[int, int]]:
        '''firings as a list of pairs time and neuron representings all firings'''
        firings = self.firings
        return [
            (time, neuron)
            for time, fired in enumerate(firings)
            for neuron in fired
        ]
    

    def score(self) -> Dict[str,Any]:
        firings = self.firings
        spikes_at_time = self._firings_to_spikes_at_time()

        CUT_OFF = 0.2
        
        cut_spikes=spikes_at_time[int(len(spikes_at_time)*CUT_OFF ):]
        
        ex_firing=cut_spikes['excitatory'].sum() / len(cut_spikes)
        in_firing=cut_spikes['inhibitory'].sum() / len(cut_spikes)
        
        total_firing = ex_firing + in_firing
        
        return dict(
            ex_firing=ex_firing,
            in_firing=in_firing,
            total_firing=total_firing
        ) 
    

    def plot_firings(self) -> None: 
        spikes_at_time = self._firings_to_spikes_at_time()
        neuron_firings = self._flatten_firings()

        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            row_width=[0.2,0.8],
            x_title='Time (ms)',
        )
        # Neurons
        # Exitatory firing
        time, neuron = zip(*filter(lambda x: x[1] <= self.Ne, neuron_firings))
        fig.append_trace(go.Scatter(
            name='Exitatory firing',
            x=time,
            y=neuron,
            mode='markers',
            marker=dict(
                size=2,
                color='blue'
            )
        ), row=1, col=1)
        
        # Inhibitory firing
        time, neuron = zip(*filter(lambda x: x[1] > self.Ne, neuron_firings))
        fig.append_trace(go.Scatter(
            name='Inhibatory firing',
            x=time,
            y=neuron,
            mode='markers',
            marker=dict(
                size=2,
                color='red'
            )
        ), row=1, col=1)

        # Firing rate
        fig.append_trace(go.Scatter(
            name='Total firing',
            y=spikes_at_time['num_fired'],
            x=spikes_at_time['time'],
            line_color='black'
        ), row=2, col=1)
        
        fig.append_trace(go.Scatter(
            name='Excitatory firing',
            y=spikes_at_time['excitatory'],
            x=spikes_at_time['time'],
            line_color='red'
        ), row=2, col=1)
        
        fig.append_trace(go.Scatter(
            name='Inhibitory',
            y=spikes_at_time['inhibitory'],
            x=spikes_at_time['time'],
            line_color='blue'
        ), row=2, col=1)

        fig.update_layout(width=700, height=700, yaxis_title='Neuron #')

        fig['layout']['yaxis2']['title'] = 'Firing rate (Hz)'

        fig.show()

    def animated_plot_firing(self) -> None:
        spikes_at_time = self._firings_to_spikes_at_time()
        neuron_firings = self._flatten_firings()

        fig_dict: Any = {
            'data': [],
            'layout': {},
            'frames': []
        }

        fig_dict['layout']['width'] = 700
        fig_dict['layout']['height'] = 700
        fig_dict['layout']['yaxis_title'] = 'Neuron #'
        fig_dict['layout']['x_title'] = 'Neuron #'
        fig_dict['layout']['shared_xaxes'] = True
        fig_dict['layout']['row_width'] = [0.2,0.8]
        # TODO fixed x,y axis based on time, n_neurons


        data = [
            {
                'name':'Excitatory Firing',
                'type':'scatter',
                'x': [],
                'y': [],
                'marker': {
                    'size':2,
                    'color':'blue'
                },
                'mode':'markers',
            },  
            {
                'name':'Inhibitory Firing',
                'type':'scatter',
                'x': [],
                'y': [],
                'marker': {
                    'size':2,
                    'color':'red'
                },
            },  
            {
                'name':'Total firing',
                'type':'scatter',
                'x': [],
                'y': [],
                'line_color':'black',
            },  
            {
                'name':'Excitatory Firing',
                'type':'scatter',
                'x': [],
                'y': [],
                'line_color':'black',
            },  
            {
                'type':'Inhibatory Firing',
                'x': [],
                'y': [],
                'line_color':'black',
                'name':'Total firing'
            },  
        ]

        frames: List[Dict] = []
        increment = 1
        for i in range(0, self.time, increment):

            frame = {
                'name':f'{i}',
                'data':[],
                'traces':[0,1,2,3,4]
            }

            data.append(dict(
                
            ))

        pass







@dataclass
class SNN:
    ge:         float = 0.5
    gi:         float = 1.0
    Ne:         int   = 800
    Ni:         int   = 200
    sparsity:   float = 1.0
    time:       int   = 1000
    thalmic_ex: float = 5
    thalmic_in: float = 2
    
    def _run_network( 
         self,
         ge:         float,
         gi:         float,
         Ne:         int,
         Ni:         int,
         sparsity:   float,
         time:       int,
         thalmic_ex: float,
         thalmic_in: float,
     ) -> List[List[int]]:
        ''' '''
        re, ri = np.random.rand( Ne ), np.random.rand( Ni )

        a=np.concatenate((0.02*np.ones(Ne), 0.02+0.08*ri), axis=0)
        b=np.concatenate((0.2*np.ones(Ne),  0.25-0.05*ri), axis=0)
        c=np.concatenate((-65+15*re**2,    -65*np.ones(Ni)), axis=0)
        d=np.concatenate(( 8-6*re**2,       2*np.ones(Ni)), axis=0)
        # Neuron connections
        S=np.concatenate((ge*np.random.rand(Ne+Ni,Ne),  -gi*np.random.rand(Ne+Ni,Ni)), axis=1)

        # Introduce sparsity to the network
        S=np.multiply( S, np.random.binomial( 1, sparsity, S.shape ) )

        # Set initial charge 
        v= -65*np.ones(Ne+Ni);
        u= b*v

        firings = []
        for t in range(0, time):
            I=np.concatenate((thalmic_ex*np.random.randn(Ne,1),thalmic_in*np.random.randn(Ni,1)), axis=0).flatten() # thalamic input
            # Get all the neurons that have fired > 30mA charge
            fired=np.where(v>=30)
            firings.append(fired[0])
            # reset all fired neurons to c
            v[fired] = c[fired]
            u[fired] = u[fired] + d[fired]
            I = I + S[:,fired[0]].sum(axis=1).flatten()

            v = v + 0.5 * (0.04*(v**2) + 5*v + 140 - u + I)
            v = v + 0.5 * (0.04*(v**2) + 5*v + 140 - u + I)
            u = u + a * (b * v - u)

        return firings

    def run_network(self) -> SNNFirings:
        '''Run network with parameters and produce result object'''
        firings = self._run_network(
            self.ge, self.gi, 
            self.Ne, self.Ni, 
            self.sparsity,
            self.time,
            self.thalmic_ex, self.thalmic_in
        )

        return SNNFirings(
            ge=self.ge, 
            gi=self.gi, 
            Ne=self.Ne, 
            Ni=self.Ni, 
            sparsity=self.sparsity,
            time=self.time,
            thalmic_ex=self.thalmic_ex,
            thalmic_in=self.thalmic_in,
            firings=firings,
        )
    
if __name__ == '__main__':
    model = SNN()
    firings = model.run_network()
    spikes_at_time = firings._firings_to_spikes_at_time()
    neuron_firings = firings._flatten_firings()

