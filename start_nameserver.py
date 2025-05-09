import Pyro4
import Pyro4.naming

def start_nameserver(ns_port=8888):
    """Starts a Pyro4 nameserver that listens on all network interfaces.

    Parameters
    ----------
    ns_port : int
        the port number for the nameserver to listen on

    Returns
    -------
    """
    Pyro4.config.SERIALIZERS_ACCEPTED = set(['pickle'])
    Pyro4.config.PICKLE_PROTOCOL_VERSION=4
    Pyro4.naming.startNSloop(host='0.0.0.0', port=ns_port)

start_nameserver(ns_port=9090)