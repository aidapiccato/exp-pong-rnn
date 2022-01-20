"""Recurrent agent that can train on batched data"""
import torch

class SimpleRNN():
    def __init__(self, hidden_size, activity_decay=0.1, input_feature_len=1,
                 output_feature_len=1):
        """Constructor.
        Args:
            hidden_size: Int. Hidden size.
            activity_decay: Float. Activity decay.
            input_feature_len: Int. Length of input features.
            output_features_len: Int Length of output features.
        """
        self._loss = torch.nn.MSELoss()

        self._hidden_size = hidden_size
        self._activity_decay = activity_decay
        self._input_feature_len = input_feature_len
        self._output_feature_len = output_feature_len

        self._activation = torch.nn.Tanh()
        self._linear = torch.nn.Linear(
            in_features=input_feature_len + hidden_size,
            out_features=hidden_size,
            bias=True,
        )
        self._decoder = torch.nn.Linear(
            in_features=hidden_size,
            out_features=output_feature_len,
            bias=True,
        )
        
    def reset(self):
        self._hiddens = None

    def step(self, timestep, env, test=False):
        obs = timestep['obs']
        batch_size = obs.size(0)

        # Initializing hidden state if it has just been reset
        if self._hiddens is None:
            self._hiddens =  torch.zeros(batch_size, self._hidden_size)

        # Apply RNN to get latent (hidden) states
        recent_hiddens = self._hiddens[-1]
        rate = self._activation(recent_hiddens)
        net_inputs = torch.cat([obs, rate], dim=1)
        net_outputs = self._linear(net_inputs)
        new_hiddens = ((1 - self._activity_decay) * recent_hiddens + net_outputs)
        self._hiddens.append(new_hiddens)



        hiddens = torch.cat([torch.unsqueeze(h, 1)
                             for h in hiddens[1:]], dim=1)

        # Apply decoder to hidden
        flat_hiddens = new_hiddens.view(batch_size, self._hidden_size)
        outputs = self._decoder.forward(flat_hiddens)
        outputs = outputs.view(batch_size, 1, self._output_feature_len)

        outs = {
            'inputs': inputs,
            'outputs': outputs,
            'hiddens': hiddens,
            'labels': data['labels'],
        }

        return outs

    