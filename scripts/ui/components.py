"""Reusable UI components: panels, buttons, tabs, charts."""

from typing import List, Optional, Sequence, Tuple

import pygame

from ui.theme import THEME, font

Color = Tuple[int, int, int]


class RectButton:
    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        key: Optional[int] = None,
        subtitle: str = "",
    ):
        self.rect = rect
        self.label = label
        self.subtitle = subtitle
        self.key = key
        self.hovered = False
        self.active = False

    def contains(self, pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)

    def draw(
        self,
        screen: pygame.Surface,
        accent: Optional[Color] = None,
        *,
        compact: bool = False,
    ) -> None:
        t = THEME
        base = t.panel_active if self.active else t.panel_hover if self.hovered else t.panel
        border = accent or (t.accent if self.active else t.border_soft)
        radius = 8 if compact else 10
        pygame.draw.rect(screen, base, self.rect, border_radius=radius)
        pygame.draw.rect(screen, border, self.rect, 2 if self.active else 1, border_radius=radius)

        text_color = t.text if self.active else t.text_secondary
        label_size = 12 if compact else (13 if len(self.label) > 14 else 14)
        surf = font(label_size, bold=self.active).render(self.label, True, text_color)
        if self.subtitle:
            screen.blit(surf, (self.rect.x + 12, self.rect.y + 8))
            sub = font(10).render(self.subtitle, True, t.text_muted)
            screen.blit(sub, (self.rect.x + 12, self.rect.y + 26))
        else:
            screen.blit(surf, surf.get_rect(center=self.rect.center))


def draw_text(
    screen: pygame.Surface,
    text: str,
    x: int,
    y: int,
    size: int = 16,
    color: Optional[Color] = None,
    bold: bool = False,
    center: bool = False,
) -> pygame.Rect:
    color = color or THEME.text
    surf = font(size, bold=bold).render(str(text), True, color)
    if center:
        rect = surf.get_rect(center=(x, y))
        screen.blit(surf, rect)
        return rect
    screen.blit(surf, (x, y))
    return surf.get_rect(topleft=(x, y))


def draw_section_header(screen: pygame.Surface, x: int, y: int, title: str) -> int:
    draw_text(screen, title.upper(), x, y, 10, THEME.text_muted, bold=True)
    return y + 18


def draw_mode_pill(
    screen: pygame.Surface,
    rect: pygame.Rect,
    label: str,
    *,
    active: bool = True,
    color: Optional[Color] = None,
) -> None:
    """Colored badge for TRAINING / AUTONOMOUS modes."""
    accent = color or (THEME.warning if active else THEME.success)
    fill = (
        max(0, accent[0] // 5),
        max(0, accent[1] // 5),
        max(0, accent[2] // 5),
    )
    pygame.draw.rect(screen, fill, rect, border_radius=rect.height // 2)
    pygame.draw.rect(screen, accent, rect, 2, border_radius=rect.height // 2)
    text_color = accent if not active else THEME.text
    draw_text(screen, label, rect.centerx, rect.centery, 11, text_color, bold=True, center=True)


def draw_epsilon_bar(
    screen: pygame.Surface,
    rect: pygame.Rect,
    epsilon: float,
    *,
    fill_color: Optional[Color] = None,
    label: str = "ε exploration",
) -> None:
    """Visual bar showing exploration rate (0–100%)."""
    fill_color = fill_color or THEME.warning
    pygame.draw.rect(screen, THEME.bg_elevated, rect, border_radius=6)
    pygame.draw.rect(screen, THEME.border_soft, rect, 1, border_radius=6)
    draw_text(screen, label, rect.x + 8, rect.y - 14, 9, THEME.text_muted, bold=True)
    inner = rect.inflate(-4, -4)
    frac = max(0.0, min(1.0, epsilon))
    if frac > 0:
        fill_w = max(4, int(inner.width * frac))
        fill_rect = pygame.Rect(inner.x, inner.y, fill_w, inner.height)
        pygame.draw.rect(screen, fill_color, fill_rect, border_radius=4)
    pct = int(round(frac * 100))
    draw_text(screen, f"{pct}%", rect.right - 8, rect.centery, 10, THEME.text_secondary, bold=True, center=True)


def draw_divider(screen: pygame.Surface, rect: pygame.Rect) -> None:
    pygame.draw.line(screen, THEME.border_soft, (rect.x, rect.y), (rect.right, rect.y))


def draw_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    title: Optional[str] = None,
    accent: Optional[Color] = None,
) -> None:
    pygame.draw.rect(screen, THEME.panel, rect, border_radius=12)
    pygame.draw.rect(screen, accent or THEME.border_soft, rect, 1, border_radius=12)
    if title:
        draw_text(screen, title, rect.x + 14, rect.y + 10, 13, THEME.text_secondary, bold=True)


def draw_stat_card(
    screen: pygame.Surface,
    rect: pygame.Rect,
    label: str,
    value: str,
    color: Optional[Color] = None,
) -> None:
    pygame.draw.rect(screen, THEME.bg_elevated, rect, border_radius=10)
    pygame.draw.rect(screen, THEME.border_soft, rect, 1, border_radius=10)
    draw_text(screen, label, rect.x + 12, rect.y + 8, 10, THEME.text_muted, bold=True)
    draw_text(screen, value, rect.x + 12, rect.y + 24, 20, color or THEME.text, bold=True)


def draw_chip(
    screen: pygame.Surface,
    rect: pygame.Rect,
    label: str,
    *,
    active: bool = False,
    accent: Optional[Color] = None,
    hotkey: str = "",
) -> None:
    color = accent or THEME.accent
    bg = THEME.panel_active if active else THEME.panel
    border = color if active else THEME.border_soft
    pygame.draw.rect(screen, bg, rect, border_radius=8)
    pygame.draw.rect(screen, border, rect, 2 if active else 1, border_radius=8)
    text_color = THEME.text if active else THEME.text_secondary
    draw_text(screen, label, rect.x + 10, rect.centery - 7, 12, text_color, bold=active)
    if hotkey:
        badge = pygame.Rect(rect.right - 26, rect.y + 6, 20, 16)
        pygame.draw.rect(screen, THEME.bg_elevated, badge, border_radius=4)
        draw_text(screen, hotkey, badge.centerx, badge.centery, 9, THEME.text_muted, center=True)


def draw_tab_bar(
    screen: pygame.Surface,
    x: int,
    y: int,
    tabs: Sequence[str],
    active_index: int,
    tab_width: int = 120,
) -> List[RectButton]:
    buttons = []
    for i, tab in enumerate(tabs):
        rect = pygame.Rect(x + i * (tab_width + 6), y, tab_width, 34)
        btn = RectButton(rect, tab)
        btn.active = i == active_index
        btn.draw(screen)
        buttons.append(btn)
    return buttons


def draw_wrapped_text(
    screen: pygame.Surface,
    text: str,
    x: int,
    y: int,
    width: int,
    size: int = 14,
    color: Optional[Color] = None,
) -> int:
    color = color or THEME.text_secondary
    f = font(size)
    words = text.split()
    line = ""
    current_y = y
    for word in words:
        test = line + word + " "
        if f.size(test)[0] <= width:
            line = test
        else:
            screen.blit(f.render(line, True, color), (x, current_y))
            current_y += size + 5
            line = word + " "
    if line:
        screen.blit(f.render(line, True, color), (x, current_y))
        current_y += size + 5
    return current_y


def draw_line_chart(
    screen: pygame.Surface,
    rect: pygame.Rect,
    values: Sequence[float],
    title: str,
    line_color: Color,
) -> None:
    draw_panel(screen, rect, title)
    chart = rect.inflate(-24, -50)
    chart.y += 36
    chart.height -= 10

    if len(values) < 2:
        draw_text(screen, "Collecting data…", chart.x, chart.centery - 8, 14, THEME.text_muted)
        return

    min_v = min(values)
    max_v = max(values)
    if min_v == max_v:
        min_v -= 1
        max_v += 1

    pygame.draw.line(screen, THEME.border_soft, (chart.left, chart.bottom), (chart.right, chart.bottom))
    pygame.draw.line(screen, THEME.border_soft, (chart.left, chart.top), (chart.left, chart.bottom))

    points = []
    for i, value in enumerate(values):
        px = chart.left + int((i / (len(values) - 1)) * chart.width)
        norm = (value - min_v) / (max_v - min_v)
        py = chart.bottom - int(norm * chart.height)
        points.append((px, py))

    if len(points) >= 2:
        pygame.draw.lines(screen, line_color, False, points, 2)
    for point in points[-12:]:
        pygame.draw.circle(screen, line_color, point, 3)

    draw_text(screen, f"Latest: {values[-1]:.1f}", rect.x + 16, rect.bottom - 22, 12, THEME.text_muted)


def draw_bar_chart(
    screen: pygame.Surface,
    rect: pygame.Rect,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    bar_color: Color,
) -> None:
    draw_panel(screen, rect, title)
    inner = rect.inflate(-20, -48)
    inner.y += 38
    if not values:
        draw_text(screen, "No data", inner.x, inner.centery, 14, THEME.text_muted)
        return

    max_v = max(values) or 1
    bar_w = max(24, inner.width // max(len(values), 1) - 10)
    x = inner.x + 8
    for label, value in zip(labels, values):
        h = int((value / max_v) * (inner.height - 28))
        bar_rect = pygame.Rect(x, inner.bottom - h - 18, bar_w, h)
        pygame.draw.rect(screen, bar_color, bar_rect, border_radius=6)
        draw_text(screen, f"{value:.0f}", x + bar_w // 2, bar_rect.top - 16, 11, THEME.text_secondary, center=True)
        short = label[:6]
        draw_text(screen, short, x + bar_w // 2, inner.bottom - 12, 10, THEME.text_muted, center=True)
        x += bar_w + 10


def draw_progress_bar(
    screen: pygame.Surface,
    rect: pygame.Rect,
    fraction: float,
    fill_color: Color,
    label: str = "",
) -> None:
    pygame.draw.rect(screen, THEME.bg_elevated, rect, border_radius=6)
    fill_w = max(0, min(rect.width, int(rect.width * fraction)))
    if fill_w > 0:
        fill_rect = pygame.Rect(rect.x, rect.y, fill_w, rect.height)
        pygame.draw.rect(screen, fill_color, fill_rect, border_radius=6)
    if label:
        draw_text(screen, label, rect.centerx, rect.centery, 11, THEME.text, center=True)
