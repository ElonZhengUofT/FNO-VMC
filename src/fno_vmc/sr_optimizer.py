import netket as nk


def make_sr_optimizer(diagshift: float = 0.01):
    """
    Create a stochastic reconfiguration optimizer.
    """
    return nk.optimizer.SR(diagshift=diagshift)