"""ms-swift external plugin entry point."""

from swift.rewards import orms

from coloreason_reward import ColoReasonCompositeReward

orms["coloreason_composite"] = ColoReasonCompositeReward
