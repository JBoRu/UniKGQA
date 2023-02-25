import sys
import logging
import numpy as np
from NSM.Agent.NSMAgent import NsmAgent, NsmAgentForPLM
logger = logging.getLogger('NSM')

def init_nsm(args, entity2id, num_relation, word2id, rel2id, device):
    if args['local_rank'] <= 0:
        logger.info("Building {}.".format("Agent"))
    if args["agent_type"] == "PLM":
        if args['local_rank'] <= 0:
            logger.info("Use NsmAgentForPLM")
        agent = NsmAgentForPLM(args, len(entity2id), num_relation, rel2id, device)
    else:
        if args['local_rank'] <= 0:
            logger.info("Use NsmAgent")
        agent = NsmAgent(args, len(entity2id), num_relation, len(word2id), device)

    return agent