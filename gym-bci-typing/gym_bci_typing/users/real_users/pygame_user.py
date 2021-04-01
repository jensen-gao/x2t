import pygame
from pygame import gfxdraw
import random
import numpy as np
from gym_bci_typing.users import User
from stable_baselines.common import TensorboardWriter
from abc import ABC


class PygameUser(User, ABC):
    """
    Interface that collects real user input to the typing environment via a Pygame interface
    :param sampling_freq: (float) Input sampling frequency (Hz).
    :param action_text_color: (Tuple(int)) Color of the text on action buttons (except for "Done" and "Next" actions).
    :param bg_color: (Tuple(int)) Color of the interface background.
    :param text_color: (Tuple(int)) Color for non-action button text.
    """
    def __init__(self, goals, sampling_freq, save_path, windowed, action_text_color=(0, 0, 0),
                 bg_color=(0, 0, 0), text_color=(255, 255, 255), **kwargs):
        super(PygameUser, self).__init__(**kwargs)
        self.goals = goals
        self.goal = None
        self.vocab = set()
        self.period = 1000 / sampling_freq  # sampling period in ms
        self.save_path = save_path
        self.text_color = text_color
        self.action_text_color = action_text_color
        self.bg_color = bg_color
        self.circle_radius = 20

        self.obs = None
        self.last_time = 0
        self.actions = None

        pygame.init()
        self.fonts = {font_size: pygame.font.Font('freesansbold.ttf', font_size) for font_size in range(2, 34, 2)}
        if windowed:
            self.screen = pygame.display.set_mode()
        else:
            self.screen = pygame.display.set_mode(flags=pygame.FULLSCREEN)

        self.width, self.height = self.screen.get_size()
        self.text_field_height = self.height / 30
        self.header_coord = (self.width / 2, self.text_field_height)
        self.button_height = self.height / 20
        self.button_width = 2 * self.button_height
        self.running = True
        self.paused = False
        self.waiting = True
        self.reward = None

        self.successes = None
        self.n_episodes = None
        self.n_steps = None
        self.baseline_estimates = None

        self.show_loading_screen()

    def setup(self, env):
        super(PygameUser, self).setup(env)
        if not self.env.no_lm:
            self.goals = [goal for goal in self.goals if all([word in self.env.lm.lm_vocab.idx2sym for word in goal])]
        for goal in self.goals:
            for word in goal:
                self.vocab.add(word)

    def reset(self):
        self.goal = random.choice(self.goals)

    def get_next_action_index(self):
        return self.env.curr_actions.index(self.goal[len(self.env.typed)])

    def show_loading_screen(self):
        self.screen.fill(self.bg_color)
        self.draw_rect_with_text('Loading...', self.text_color, self.width, self.height,
                                 center=(self.width / 2, self.height / 2))
        pygame.display.flip()
        self.get_event(can_pause=False)

    def run(self, total_timesteps, callback=None, tb_log_name='log', mode='m', disable_learning=False):
        """
        Runs the interface.
        """
        self.model.setup_learn(callback=callback)
        self.env.reset()

        self.main_loop(tb_log_name=tb_log_name, mode=mode, total_timesteps=total_timesteps,
                       disable_learning=disable_learning)
        pygame.quit()

        metrics = {'accuracy': self.successes / self.n_steps}
        return metrics, self.baseline_estimates

    def main_loop(self, tb_log_name, mode, total_timesteps, disable_learning):
        """
        Main loop of the typing interface
        """
        self.reset_step()
        self.screen.fill(self.bg_color)
        self.reset_interface()
        self.reset_actions()
        pygame.display.flip()

        self.successes = 0
        self.n_episodes = 0
        self.n_steps = 0
        started = False

        self.baseline_estimates = [[] for _ in range(self.env.action_space.n)]

        with TensorboardWriter(self.model.graph, self.model.tensorboard_log, tb_log_name, self.model.new_tb_log) \
                as writer:
            while self.running:
                self.get_event()
                if not self.paused:
                    if self.waiting:
                        continue

                    if not started:
                        self.draw_rect_with_text('', self.text_color, 1.7 * self.env.radius, self.text_field_height,
                                                 center=(self.width / 2, self.height / 2))
                        pygame.display.flip()
                        if self.obs is not None:
                            # TODO: last word per sentence is not processed
                            self.model.step(self.obs, act, self.reward, done, info, target, writer=writer,
                                            learn=mode != 'b' and (not disable_learning))
                        self.obs = []
                        started = True

                    curr_time = pygame.time.get_ticks()
                    if curr_time > self.last_time + self.period:
                        self.last_time = curr_time
                        user_input = self.get_input()
                        self.obs.append(user_input)

                        # gets current action prediction distribution
                        if len(self.obs) == self.n_samples:
                            if mode == 'b':
                                baseline_proba, estimate = self.baseline(self.obs)
                                proba = baseline_proba / self.baseline_temp
                            elif mode == 'l':
                                proba = self.predict(self.obs)
                            else:
                                proba = self.predict(self.obs)
                                baseline_proba, estimate = self.baseline(self.obs)
                                proba = proba + (baseline_proba / self.baseline_temp)

                            if self.boltzmann_exploration:
                                proba += np.random.gumbel(proba.shape)
                            act = np.argmax(proba)

                            self.n_steps += 1
                            _, _, done, info = self.env.step(act)
                            target = info['target']

                            if info['correct']:
                                self.successes += 1

                            if mode != 'l' and estimate is not None:
                                default_correct = np.argmax(baseline_proba) == target
                                self.baseline_estimates[target].append((estimate, default_correct))

                            if done:
                                self.env.reset()
                                self.obs = None
                                self.n_episodes += 1

                            if self.n_steps >= total_timesteps:
                                break

                            self.reset_step()
                            self.screen.fill(self.bg_color)
                            self.reset_interface()
                            self.reset_actions()
                            pygame.display.flip()
                            started = False
                            self.reward = 1

    def reset_step(self):
        """
        Prepares for the next action selection step, should be called after the end of an action selection step
        (either when an action was taken in the environment, or when the interface is unpaused).
        """
        self.last_time = - (self.period + 1)
        self.waiting = True

    def reset_interface(self):
        """
        Resets the main interface.
        """
        rect = self.draw_rect_with_text('Goal: ' + ' '.join(self.goal), self.text_color, self.width,
                                        self.text_field_height, center=self.header_coord)
        text = 'Typed: '
        if len(self.env.typed) > 1:
            text += ' '.join(self.env.typed[:-1]) + ' '

        rect = self.draw_rect_with_text(text, self.text_color, self.width,
                                        self.text_field_height, left=rect.left, top=rect.bottom)
        if len(self.env.typed) > 0:
            color = self.text_color if self.env.typed[-1] == self.goal[len(self.env.typed) - 1] else (255, 0, 0)
            self.draw_rect_with_text(self.env.typed[-1], color, self.width,
                                     self.text_field_height, left=rect.right, top=rect.top)
        self.draw_rect_with_text('Next Word: ' + self.goal[len(self.env.typed)], self.text_color, self.width,
                                 self.text_field_height,
                                 center=(self.header_coord[0], self.header_coord[1] + 2 * self.text_field_height))
        self.draw_rect_with_text('Episodes: ' + str(self.n_episodes), self.text_color, 2 * self.button_width,
                                 self.text_field_height, top=self.height - 3 * self.text_field_height, left=20)
        self.draw_rect_with_text('Steps: ' + str(self.n_steps), self.text_color, 2 * self.button_width,
                                 self.text_field_height, top=self.height - 2 * self.text_field_height, left=20)
        self.draw_rect_with_text('Press SPACE to start recording', self.text_color,
                                 self.width, self.text_field_height,
                                 center=(self.width / 2, self.height / 2))

    def reset_actions(self):
        """
        Updates the action buttons
        """
        self.actions = self.env.curr_actions

        for i, action in enumerate(self.actions):
            coord = self.env.uncenter_coord(self.env.action_coords[i])
            text_color = self.action_text_color
            self.draw_rect_with_text(action, text_color, self.button_width, self.button_height, (255, 255, 255),
                                     center=coord)

    def pause(self):
        """
        Displays a pause screen.
        """
        self.screen.fill(self.bg_color)
        self.reset_interface()
        self.draw_rect_with_text("Paused", self.text_color, self.width, self.height,
                                 center=(self.width / 2, self.height / 2))
        pygame.display.flip()

    def draw_rect_with_text(self, text, text_color, width, height, rect_color=None, font_size=32, center=None,
                            left=0, top=0):
        """
        Creates a rectangle with the provided width and height at the provided center location. Draws the rectangle
        with rect_color if it is not None. Writes text with the desired font color at the center of the rectangle.
        The font size will shrink until it can fit within the rectangle in a single line.
        """
        while font_size > 2:
            font = self.fonts[font_size]
            req_width, req_height = font.size(text)
            if req_width <= width and req_height <= height:
                break
            font_size -= 2
        rect = pygame.Rect(left, top, width, height)
        if center is not None:
            rect.center = center
        if rect_color is None:
            rect_color = self.bg_color
        pygame.draw.rect(self.screen, rect_color, rect)
        text_img = font.render(text, True, text_color)
        if center is not None:
            text_rect = text_img.get_rect(center=center)
        else:
            text_rect = text_img.get_rect(left=left, top=top)
        self.screen.blit(text_img, text_rect)
        return text_rect

    def draw_circle_with_text(self, text, text_color, center, radius, circle_color=None, font_size=32):
        """
        Creates a circle with the provided width and height at the provided center location. Draws the circle
        with circle_color if it is not None. Writes text with the desired font color at the center of the circle.
        The font size will shrink until it can fit within the circle.
        """
        font = self.fonts[font_size]
        while font_size > 2:
            req_width, req_height = font.size(text)
            if req_width <= 2 * radius and req_height <= 2 * radius:
                break
            font_size -= 2
            font = self.fonts[font_size]
        center = np.round(center).astype(int)
        if circle_color is None:
            circle_color = self.bg_color
        gfxdraw.aacircle(self.screen, center[0], center[1], radius, circle_color)
        gfxdraw.filled_circle(self.screen, center[0], center[1], radius, circle_color)
        text_img = font.render(text, True, text_color)
        text_rect = text_img.get_rect(center=center)
        self.screen.blit(text_img, text_rect)

    def get_event(self, can_pause=True):
        """
        Gets and processes all events at the current timestep in the interface. Must be called to progress to the next
        timestep.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_p and can_pause:
                    if self.paused:
                        self.reset_step()
                        self.screen.fill(self.bg_color)
                        self.reset_interface()
                        self.reset_actions()
                        pygame.display.flip()
                    else:
                        self.pause()
                    self.paused = not self.paused
                elif event.key == pygame.K_SPACE:
                    self.waiting = False
                elif event.key == pygame.K_BACKSPACE:
                    if len(self.env.typed) > 0 and self.reward != 0 and self.waiting:
                        self.env.undo()
                        self.screen.fill(self.bg_color)
                        self.reset_interface()
                        self.reset_actions()
                        pygame.display.flip()
                        self.reward = 0
