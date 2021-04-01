from gym.envs.registration import register

register(
    id='RadialTyping-v0',
    entry_point='gym_bci_typing.envs:RadialTypingEnv',
)

register(
    id='CharTyping-v0',
    entry_point='gym_bci_typing.envs:CharTypingEnv',
)
