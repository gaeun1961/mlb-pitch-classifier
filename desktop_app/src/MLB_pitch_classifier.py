# MLB_pitch_classifier.py - 구종 분류기 GUI (tkinter)

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from predict import predict
from data_loader import FEATURE_COLS

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

PITCH_NAMES = {
    'FF': '포심 패스트볼', 'SI': '싱커',    'SL': '슬라이더',
    'CU': '커브',          'CH': '체인지업', 'FC': '커터', 'FS': '스플리터',
}
PITCH_COLORS = {
    'FF': '#E53935', 'SI': '#FB8C00', 'SL': '#8E24AA',
    'CU': '#1E88E5', 'CH': '#43A047', 'FC': '#F4511E', 'FS': '#00ACC1',
}
FIELD_LABELS = {
    'release_speed'    : '구속 (mph)',
    'release_spin_rate': '회전수 (rpm)',
    'release_extension': '릴리스 익스텐션 (ft)',
    'release_pos_x'    : '릴리스 위치 X (ft)',
    'release_pos_z'    : '릴리스 위치 Z (ft)',
    'pfx_x'            : '수평 변화량 (ft)',
    'pfx_z'            : '수직 변화량 (ft)',
    'plate_x'          : '홈플레이트 X (ft)',
    'plate_z'          : '홈플레이트 Z (ft)',
    'vx0'              : '속도 벡터 X',
    'vy0'              : '속도 벡터 Y',
    'vz0'              : '속도 벡터 Z',
    'ax'               : '가속도 X',
    'ay'               : '가속도 Y',
    'az'               : '가속도 Z',
    'effective_speed'  : '체감 구속 (mph)',
    'spin_axis'        : '스핀 축 (°)',
}
SAMPLE_VALUES = {
    'release_speed'    : '94.5',   'release_spin_rate': '2280.0',
    'release_extension': '6.2',    'release_pos_x'    : '-1.5',
    'release_pos_z'    : '6.1',    'pfx_x'            : '0.8',
    'pfx_z'            : '1.2',    'plate_x'          : '0.3',
    'plate_z'          : '2.8',    'vx0'              : '5.2',
    'vy0'              : '-138.0', 'vz0'              : '-5.1',
    'ax'               : '8.3',    'ay'               : '28.5',
    'az'               : '-14.2',  'effective_speed'  : '93.1',
    'spin_axis'        : '210.0',
}
DISPLAY_COLS = ['player_name', 'p_throws', 'release_speed', 'release_spin_rate',
                'effective_speed', 'game_date', 'inning']
PITCH_DIST = 60.5

# 슬라이더: (col, 라벨, 설명, min, max, resolution)
SLIDERS = [
    ('release_speed',    '구속 (mph)',
     '빠를수록 FF/FC 계열',             60.0, 105.0,  0.5),
    ('release_spin_rate','회전수 (rpm)',
     '높을수록 포심/커터 계열',        1500.0,3500.0, 10.0),
    ('az',  '수직 가속도 az',
     '양수=포심(떠오름), 음수=싱커(가라앉음)', -50.0, 15.0, 0.5),
    ('ax',  '수평 가속도 ax',
     '양수=우완 슬라이더, 음수=우완 커브', -30.0, 30.0, 0.5),
]


class CSVSelectDialog:
    def __init__(self, parent, df):
        self.result = None
        self.win    = tk.Toplevel(parent)
        self.win.title('투구 선택 (구종을 맞혀보세요!)')
        self.win.grab_set()
        self._build(df)

    def _build(self, df):
        tk.Label(self.win, text='투구를 선택하고 구종을 예측해보세요!',
                 font=('Arial', 11, 'bold')).pack(pady=8)
        show_cols = [c for c in DISPLAY_COLS if c in df.columns]
        frame = tk.Frame(self.win)
        frame.pack(padx=10, pady=4, fill='both', expand=True)
        sb = ttk.Scrollbar(frame, orient='vertical')
        sb.pack(side='right', fill='y')
        self.tree = ttk.Treeview(frame, columns=show_cols,
                                  show='headings', height=15,
                                  yscrollcommand=sb.set)
        sb.config(command=self.tree.yview)
        col_widths = {'player_name': 140, 'release_speed': 90,
                      'release_spin_rate': 100, 'effective_speed': 100,
                      'game_date': 100, 'inning': 60}
        for c in show_cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=col_widths.get(c, 100), anchor='center')
        self.index_map = {}
        for idx, _ in df[FEATURE_COLS].dropna().head(300).iterrows():
            vals = [df.loc[idx, c] if c in df.columns else '-' for c in show_cols]
            iid  = self.tree.insert('', 'end', values=vals)
            self.index_map[iid] = idx
        self.tree.pack(fill='both', expand=True)
        tk.Button(self.win, text='이 투구로 예측하기', command=self._select,
                  bg='#2196F3', fg='white',
                  font=('Arial', 10, 'bold'), width=18).pack(pady=8)

    def _select(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning('경고', '투구를 선택하세요.', parent=self.win)
            return
        self.result = self.index_map[selected[0]]
        self.win.destroy()


class PitchClassifierApp:

    def __init__(self, root):
        self.root        = root
        self.root.title('MLB 구종 분류기')
        self.root.resizable(True, True)
        self.entries1    = {}
        self.slider_vals = {}
        self.slider_vars = {}
        self.val_labels  = {}
        self.csv_df      = None
        self.p_throws    = 'R'
        self.throws_var  = tk.StringVar(value='R')  # 미리 생성
        self._build_ui()

    def _build_ui(self):
        tk.Label(self.root, text='MLB Pitch Classifier',
                 font=('Arial', 14, 'bold')).pack(pady=(4, 2))

        nb = ttk.Notebook(self.root)
        nb.pack(fill='both', expand=True, padx=10, pady=4)

        tab1 = tk.Frame(nb)
        tab2 = tk.Frame(nb)
        nb.add(tab1, text='  수치 예측  ')
        nb.add(tab2, text='  슬라이더 예측  ')

        self._build_tab_numeric(tab1)
        self._build_tab_slider(tab2)

    def _make_canvas(self, parent, figsize):
        fig = Figure(figsize=figsize, dpi=90)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        return fig, canvas

    # ── 탭1: 수치 예측 ───────────────────────────────────
    def _build_tab_numeric(self, parent):
        top = tk.Frame(parent)
        top.pack(fill='x', padx=6, pady=6)

        inp = tk.Frame(top)
        inp.pack(side='left', anchor='n')

        bf = tk.Frame(inp)
        bf.grid(row=0, column=0, columnspan=4, pady=(4, 4))
        tk.Button(bf, text='CSV 불러오기', command=self._load_csv,
                  width=13).pack(side='left', padx=4)
        tk.Button(bf, text='투구 선택', command=self._open_select,
                  width=13).pack(side='left', padx=4)

        for i, col in enumerate(FEATURE_COLS):
            r = i // 2 + 1
            c = (i % 2) * 2
            tk.Label(inp, text=FIELD_LABELS[col],
                     anchor='w', width=20).grid(
                     row=r, column=c, padx=(6, 2), pady=2, sticky='w')
            entry = tk.Entry(inp, width=9)
            entry.insert(0, SAMPLE_VALUES[col])
            entry.grid(row=r, column=c + 1, padx=(0, 8), pady=2)
            self.entries1[col] = entry

        btn_row = len(FEATURE_COLS) // 2 + 2
        tk.Button(inp, text='구종 예측', command=self._predict_numeric,
                  bg='#2196F3', fg='white',
                  font=('Arial', 10, 'bold'), width=11).grid(
                  row=btn_row, column=0, columnspan=2, pady=6, padx=6, sticky='w')
        tk.Button(inp, text='초기화', command=self._reset,
                  width=11).grid(row=btn_row, column=2, columnspan=2, pady=6)
        self.result_label1 = tk.Label(inp, text='',
                                      font=('Arial', 11, 'bold'), fg='#1565C0')
        self.result_label1.grid(row=btn_row + 1, column=0, columnspan=4, pady=2)

        hand_frame1 = tk.Frame(inp)
        hand_frame1.grid(row=btn_row + 2, column=0, columnspan=4, pady=(2, 4))
        tk.Label(hand_frame1, text='투구 손:', font=('Arial', 9)).pack(side='left')
        tk.Radiobutton(hand_frame1, text='우완 (R)', variable=self.throws_var,
                       value='R', font=('Arial', 9),
                       command=self._on_throws_change).pack(side='left', padx=4)
        tk.Radiobutton(hand_frame1, text='좌완 (L)', variable=self.throws_var,
                       value='L', font=('Arial', 9),
                       command=self._on_throws_change).pack(side='left', padx=4)

        bar = tk.Frame(top)
        bar.pack(side='left', padx=(10, 4), anchor='n')
        self.fig_bar1, self.canvas_bar1 = self._make_canvas(bar, (3.0, 3.8))
        self.ax_bar1 = self.fig_bar1.add_subplot(111)
        self.canvas_bar1.get_tk_widget().pack()
        self._draw_bar(self.ax_bar1, self.canvas_bar1, None, None)

        bot = tk.Frame(parent)
        bot.pack(fill='x', padx=6, pady=(0, 6))
        self.fig1, self.canvas1 = self._make_canvas(bot, (10, 3.4))
        self.ax1_side = self.fig1.add_subplot(131)
        self.ax1_top  = self.fig1.add_subplot(132)
        self.ax1_zone = self.fig1.add_subplot(133)
        self.fig1.tight_layout(pad=1.6)
        self.canvas1.get_tk_widget().pack()
        self._draw_views(self.ax1_side, self.ax1_top, self.ax1_zone,
                         self.canvas1, None, None)

    # ── 탭2: 슬라이더 예측 ───────────────────────────────
    def _build_tab_slider(self, parent):

        # ── 상단: 슬라이더 영역 ─────────────────────────
        sl_frame = tk.LabelFrame(parent, text='궤적 슬라이더',
                                  font=('Arial', 9, 'bold'), fg='#333',
                                  padx=10, pady=6)
        sl_frame.pack(fill='x', padx=10, pady=(4, 2))

        for i, (col, label, desc, mn, mx, res) in enumerate(SLIDERS):
            r = i // 2
            c = (i % 2) * 4

            tk.Label(sl_frame, text=label,
                     font=('Arial', 9, 'bold'), width=18,
                     anchor='e').grid(row=r*2,   column=c, padx=(4, 2), sticky='e')
            desc_lbl = tk.Label(sl_frame, text=desc,
                     font=('Arial', 8), fg='#555', width=35,
                     anchor='w')
            desc_lbl.grid(row=r*2+1, column=c, padx=(4, 2), sticky='w')
            if col == 'ax':
                self.ax_desc_label = desc_lbl

            var = tk.DoubleVar(value=float(SAMPLE_VALUES.get(col, 0)))
            self.slider_vars[col] = var
            sl = tk.Scale(sl_frame, variable=var, from_=mn, to=mx,
                          resolution=res, orient='horizontal', length=280,
                          showvalue=False,
                          command=lambda v, c=col: self._on_slider(c))
            sl.grid(row=r*2, column=c+1, rowspan=2, padx=4, pady=2)

            val_lbl = tk.Label(sl_frame, textvariable=var,
                                font=('Arial', 10, 'bold'), fg='#1565C0', width=7)
            val_lbl.grid(row=r*2, column=c+2, rowspan=2, padx=(2, 12))
            self.val_labels[col] = val_lbl

        # ── 중단: 그래프(좌) + 스트라이크존/막대(우) ─────
        mid = tk.Frame(parent)
        mid.pack(fill='x', padx=10, pady=2)

        left = tk.Frame(mid)
        left.pack(side='left', anchor='n')

        self.fig2, self.canvas2 = self._make_canvas(left, (7.0, 3.5))
        self.ax2_side = self.fig2.add_subplot(121)
        self.ax2_top  = self.fig2.add_subplot(122)
        self.fig2.tight_layout(pad=1.8)
        self.canvas2.get_tk_widget().pack()
        self.canvas2.mpl_connect('button_press_event', self._on_zone_click)

        right = tk.Frame(mid)
        right.pack(side='left', padx=(8, 0), anchor='n')

        charts = tk.Frame(right)
        charts.pack()

        zone_f = tk.Frame(charts)
        zone_f.pack(side='left', anchor='n', padx=(0, 4))
        self.fig_zone, self.canvas_zone = self._make_canvas(zone_f, (2.6, 3.5))
        self.ax2_zone = self.fig_zone.add_subplot(111)
        self.canvas_zone.get_tk_widget().pack()
        self.canvas_zone.mpl_connect('button_press_event', self._on_zone_click)

        bar_f = tk.Frame(charts)
        bar_f.pack(side='left', anchor='n')
        self.fig_bar2, self.canvas_bar2 = self._make_canvas(bar_f, (3.2, 3.5))
        self.ax_bar2 = self.fig_bar2.add_subplot(111)
        self.canvas_bar2.get_tk_widget().pack()
        self._draw_bar(self.ax_bar2, self.canvas_bar2, None, None)

        # ── 하단: 좌완/우완 + 예측 결과 ─────────────────────
        self.result_label2 = tk.Label(parent, text='',
                                      font=('Arial', 14, 'bold'), fg='#1565C0')
        self.result_label2.pack(pady=6)

        btn_frame = tk.Frame(parent)
        btn_frame.pack(pady=(0, 8))

        hand_frame = tk.Frame(btn_frame)
        hand_frame.pack(side='left', padx=12)
        tk.Label(hand_frame, text='투구 손:', font=('Arial', 9)).pack(side='left')
        tk.Radiobutton(hand_frame, text='우완 (R)', variable=self.throws_var,
                       value='R', font=('Arial', 9),
                       command=self._on_throws_change).pack(side='left', padx=4)
        tk.Radiobutton(hand_frame, text='좌완 (L)', variable=self.throws_var,
                       value='L', font=('Arial', 9),
                       command=self._on_throws_change).pack(side='left', padx=4)

        tk.Button(btn_frame, text='초기화', command=self._reset_draw,
                  width=12).pack(side='left', padx=12)

        # ── 구종별 가이드 ─────────────────────────────────
        guide_frame = tk.LabelFrame(parent, text='구종별 만들기 가이드',
                                     font=('Arial', 9, 'bold'), fg='#333',
                                     padx=8, pady=2)
        guide_frame.pack(fill='x', padx=10, pady=(2, 4))

        # (코드, 이름, 구속 설명, az 설명, ax 설명-우완 기준, ax 설명-좌완 기준)
        guide_rows = [
            ('FF', '포심패스트볼', '구속↑(93+)',       'az 양수 크게(떠오름)',
             'ax 0 근처', 'ax 0 근처'),
            ('SI', '싱커',         '구속↑(92+)',       'az 음수(살짝 가라앉음)',
             'ax 양수(몸쪽으로 휨)', 'ax 음수(몸쪽으로 휨)'),
            ('SL', '슬라이더',     '구속중간(82-88)',  'az 0~음수 약간',
             'ax 양수 크게(바깥쪽으로 휨)', 'ax 음수 크게(바깥쪽으로 휨)'),
            ('CU', '커브',         '구속↓(75-82)',     'az 음수 크게(뚝 떨어짐)',
             'ax 음수(바깥쪽 반대로 휨)', 'ax 양수(바깥쪽 반대로 휨)'),
            ('CH', '체인지업',     '구속↓(80-87)',     'az 음수 약간',
             'ax 양수 약간', 'ax 음수 약간'),
            ('FC', '커터',         '구속↑(88-94)',     'az 0 근처',
             'ax 음수 약간(살짝 반대로 휨)', 'ax 양수 약간(살짝 반대로 휨)'),
            ('FS', '스플리터',     '구속중간(83-90)',  'az 음수 매우 크게(급격히 떨어짐)',
             'ax 0 근처', 'ax 0 근처'),
        ]
        self.guide_ax_labels = []
        for r, (code, name, spd_desc, az_desc, ax_r, ax_l) in enumerate(guide_rows):
            tk.Label(guide_frame, text=code, font=('Arial', 9, 'bold'),
                     fg='#1565C0', width=4, anchor='w').grid(row=r, column=0, sticky='w', padx=(2,8), pady=1)
            tk.Label(guide_frame, text=name, font=('Arial', 9),
                     fg='#333', width=12, anchor='w').grid(row=r, column=1, sticky='w', padx=(0,12), pady=1)
            tk.Label(guide_frame, text=spd_desc, font=('Arial', 9),
                     fg='#444', width=18, anchor='w').grid(row=r, column=2, sticky='w', padx=(0,16), pady=1)
            tk.Label(guide_frame, text=az_desc, font=('Arial', 9),
                     fg='#444', width=26, anchor='w').grid(row=r, column=3, sticky='w', padx=(0,16), pady=1)
            ax_lbl = tk.Label(guide_frame, text=ax_r, font=('Arial', 9),
                     fg='#444', width=30, anchor='w')
            ax_lbl.grid(row=r, column=4, sticky='w', pady=1)
            self.guide_ax_labels.append((ax_lbl, ax_r, ax_l))

        self._sync_and_predict()

    # ── 슬라이더 이벤트 ──────────────────────────────────

    def _on_throws_change(self):
        self.p_throws = self.throws_var.get()
        if hasattr(self, 'ax_desc_label'):
            if self.p_throws == 'L':
                self.ax_desc_label.config(
                    text='양수=좌완 슬라이더, 음수=좌완 커브')
            else:
                self.ax_desc_label.config(
                    text='양수=우완 슬라이더, 음수=우완 커브')
        if hasattr(self, 'guide_ax_labels'):
            for ax_lbl, ax_r, ax_l in self.guide_ax_labels:
                ax_lbl.config(text=ax_l if self.p_throws == 'L' else ax_r)
        self._sync_and_predict()

    def _on_slider(self, col):
        self._sync_and_predict()

    def _sync_and_predict(self):
        for col, _, _, _, _, _ in SLIDERS:
            self.slider_vals[col] = self.slider_vars[col].get()

        for col in FEATURE_COLS:
            if col not in self.slider_vals:
                self.slider_vals[col] = float(SAMPLE_VALUES[col])

        # 릴리스 위치 X: 좌완이면 양수, 우완이면 음수
        base_rx = abs(float(SAMPLE_VALUES['release_pos_x']))
        self.slider_vals['release_pos_x'] = base_rx if self.p_throws == 'L' else -base_rx

        # vz0, vx0를 az, ax에 비례시켜야 pfx_z/pfx_x가 한 클래스로 쏠리지 않음
        az_val = self.slider_vars['az'].get()
        ax_val = self.slider_vars['ax'].get()
        self.slider_vals['vz0'] = round(-3.0 - 0.15 * az_val, 2)
        self.slider_vals['vx0'] = round(0.20 * ax_val, 2)

        spd = self.slider_vals['release_speed']
        vz0 = self.slider_vals['vz0']
        az  = self.slider_vals['az']
        vx0 = self.slider_vals['vx0']
        ax_ = self.slider_vals['ax']
        ext = self.slider_vals['release_extension']
        rz  = self.slider_vals['release_pos_z']
        rx  = self.slider_vals['release_pos_x']

        vy0  = -(spd * 1.467)
        dist = PITCH_DIST - ext
        t    = dist / max(abs(vy0), 1)

        self.slider_vals['vy0']             = round(vy0, 2)
        self.slider_vals['effective_speed'] = round(spd * 0.984, 2)
        self.slider_vals['pfx_z']           = round(vz0 * t + 0.5 * az  * t**2, 3)
        self.slider_vals['pfx_x']           = round(vx0 * t + 0.5 * ax_ * t**2, 3)
        self.slider_vals['plate_z']         = round(float(np.clip(rz + vz0*t + 0.5*az*t**2,  0, 5)), 3)
        self.slider_vals['plate_x']         = round(float(np.clip(rx + vx0*t + 0.5*ax_*t**2, -2, 2)), 3)

        inp = {col: self.slider_vals[col] for col in FEATURE_COLS}

        label, conf, proba_dict = predict(inp)
        kor = PITCH_NAMES.get(label, label)
        self.result_label2.config(
            text=f'예측 구종: {label} ({kor})  {conf*100:.1f}%')

        self._draw_side(self.ax2_side, inp, label)
        self._draw_top(self.ax2_top,   inp, label)
        self.canvas2.draw()

        self._draw_zone(self.ax2_zone, inp, label)
        self.canvas_zone.draw()

        self._draw_bar(self.ax_bar2, self.canvas_bar2, proba_dict, label)

    def _on_zone_click(self, event):
        if event.inaxes == self.ax2_zone and event.xdata and event.ydata:
            self.slider_vals['plate_x'] = round(float(np.clip(event.xdata, -2, 2)), 3)
            self.slider_vals['plate_z'] = round(float(np.clip(event.ydata, 0, 5)), 3)
            inp = {col: self.slider_vals[col] for col in FEATURE_COLS}
            label, conf, proba_dict = predict(inp)
            kor = PITCH_NAMES.get(label, label)
            self.result_label2.config(
                text=f'예측 구종: {label} ({kor})  {conf*100:.1f}%')
            self._draw_zone(self.ax2_zone, inp, label)
            self.canvas_zone.draw()
            self._draw_bar(self.ax_bar2, self.canvas_bar2, proba_dict, label)

    # ── 공통 뷰 그리기 ───────────────────────────────────

    def _draw_views(self, ax_side, ax_top, ax_zone, canvas, inp, label):
        self._draw_side(ax_side, inp, label)
        self._draw_top(ax_top,   inp, label)
        self._draw_zone(ax_zone, inp, label)
        canvas.draw()

    def _draw_side(self, ax, inp, label):
        ax.clear()
        ax.set_title('측면 뷰', fontsize=9)
        ax.set_xlabel('거리 (ft)', fontsize=7)
        ax.set_ylabel('높이 (ft)', fontsize=7)
        ax.set_xlim(0, PITCH_DIST)
        ax.set_ylim(0, 8)
        ax.tick_params(labelsize=6)
        ax.axvline(x=PITCH_DIST, color='gray', linestyle='--', linewidth=0.8)
        ax.text(PITCH_DIST - 2, 0.3, '홈', fontsize=6, color='gray')
        ax.set_facecolor('#FAFAFA')

        tr = ax.transAxes
        sc = '#D8D8D8'

        ax.add_patch(patches.Circle((0.055, 0.84), 0.058,
            transform=tr, color=sc, zorder=0, clip_on=False))
        ax.add_patch(patches.FancyBboxPatch((0.018, 0.52), 0.074, 0.24,
            boxstyle="round,pad=0.018",
            transform=tr, color=sc, zorder=0, clip_on=False))
        ax.add_patch(patches.FancyBboxPatch((0.018, 0.00), 0.030, 0.50,
            boxstyle="round,pad=0.010",
            transform=tr, color=sc, zorder=0, clip_on=False))
        ax.add_patch(patches.FancyBboxPatch((0.060, 0.00), 0.030, 0.50,
            boxstyle="round,pad=0.010",
            transform=tr, color=sc, zorder=0, clip_on=False))

        if inp is None:
            return
        vy0 = inp['vy0']; vz0 = inp['vz0']
        az  = inp['az'];  rz  = inp['release_pos_z']
        ext = inp['release_extension']
        dist   = PITCH_DIST - ext
        t_vals = np.linspace(0, dist / max(abs(vy0), 1), 50)
        z = rz + vz0 * t_vals + 0.5 * az * t_vals ** 2
        x = ext + abs(vy0) * t_vals
        color = PITCH_COLORS.get(label, '#2196F3')
        ax.plot(x, z, color=color, linewidth=2.5)
        ax.plot(x[0],  z[0],  'o', color=color, markersize=5)
        ax.plot(x[-1], z[-1], '*', color=color, markersize=9)
        if label:
            ax.legend([f'{label}  {PITCH_NAMES.get(label,"")}'], fontsize=7)

    def _draw_top(self, ax, inp, label):
        ax.clear()
        ax.set_title('상단 뷰', fontsize=9)
        ax.set_xlabel('거리 (ft)', fontsize=7)
        ax.set_ylabel('좌우 (ft)', fontsize=7)
        ax.set_xlim(0, PITCH_DIST)
        ax.set_ylim(-3, 3)
        ax.tick_params(labelsize=6)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(x=PITCH_DIST, color='gray', linestyle='--', linewidth=0.8)
        ax.set_facecolor('#FAFAFA')

        tr = ax.transAxes
        sc = '#D8D8D8'

        ls = patches.Ellipse((0.05, 0.63), 0.07, 0.16,
            transform=tr, color=sc, zorder=0, clip_on=False)
        ls.set_transform(tr); ax.add_patch(ls)
        rs = patches.Ellipse((0.05, 0.37), 0.07, 0.16,
            transform=tr, color=sc, zorder=0, clip_on=False)
        rs.set_transform(tr); ax.add_patch(rs)
        hd = patches.Circle((0.05, 0.50), 0.065,
            transform=tr, color='#DCDCDC', zorder=1, clip_on=False)
        hd.set_transform(tr); ax.add_patch(hd)

        if inp is None:
            return
        vy0 = inp['vy0']
        ax_ = inp['ax'];  rx  = inp['release_pos_x']
        ext = inp['release_extension']
        dist   = PITCH_DIST - ext
        t_vals = np.linspace(0, dist / max(abs(vy0), 1), 50)
        # Statcast ax 부호는 포수 시점 기준이라 상단뷰 y축과 반대이므로 반전
        x_pos  = rx + 0.5 * (-ax_) * t_vals ** 2
        y_pos  = ext + abs(vy0) * t_vals
        color = PITCH_COLORS.get(label, '#2196F3')
        ax.plot(y_pos, x_pos, color=color, linewidth=2.5)
        ax.plot(y_pos[0],  x_pos[0],  'o', color=color, markersize=5)
        ax.plot(y_pos[-1], x_pos[-1], '*', color=color, markersize=9)

    def _draw_zone(self, ax, inp, label):
        ax.clear()
        ax.set_title('스트라이크존', fontsize=9)
        ax.set_xlabel('좌우 (ft)', fontsize=7)
        ax.set_ylabel('높이 (ft)', fontsize=7)
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 5)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(labelsize=6)
        ax.set_facecolor('#FAFAFA')
        zone = patches.Rectangle(
            (-0.708, 1.5), 1.417, 2.0,
            linewidth=2, edgecolor='#333', facecolor='#E3F2FD', zorder=2)
        ax.add_patch(zone)
        ax.text(0, 3.65, 'Strike Zone', ha='center', fontsize=6, color='#555')
        if inp:
            color = PITCH_COLORS.get(label, '#2196F3')
            ax.plot(inp['plate_x'], inp['plate_z'], 'o',
                    color=color, markersize=10, zorder=3)

    def _draw_bar(self, ax, canvas, proba_dict, label):
        ax.clear()
        ax.set_title('구종별 예측 확률', fontsize=9)
        ax.set_facecolor('#FAFAFA')
        if proba_dict is None:
            ax.set_xlim(0, 100)
            canvas.draw()
            return
        labels = list(proba_dict.keys())
        values = [proba_dict[k] * 100 for k in labels]
        colors = [PITCH_COLORS.get(k, '#90CAF9') for k in labels]
        bars   = ax.barh(labels, values, color=colors)
        ax.set_xlabel('%', fontsize=7)
        ax.set_xlim(0, 115)
        ax.tick_params(labelsize=7)
        for bar, v in zip(bars, values):
            if v > 1:
                ax.text(v + 1, bar.get_y() + bar.get_height() / 2,
                        f'{v:.1f}%', va='center', fontsize=6)
        canvas.draw()

    # ── 수치 예측 ────────────────────────────────────────
    def _predict_numeric(self):
        try:
            inp = {col: float(self.entries1[col].get()) for col in FEATURE_COLS}
        except ValueError:
            messagebox.showerror('입력 오류', '모든 항목에 숫자를 입력하세요.')
            return
        label, conf, proba_dict = predict(inp)
        kor = PITCH_NAMES.get(label, label)
        self.result_label1.config(text=f'예측 구종: {label} ({kor})  {conf*100:.1f}%')
        self._draw_views(self.ax1_side, self.ax1_top, self.ax1_zone,
                         self.canvas1, inp, label)
        self._draw_bar(self.ax_bar1, self.canvas_bar1, proba_dict, label)

    # ── CSV 관련 ─────────────────────────────────────────
    def _load_csv(self):
        path = filedialog.askopenfilename(
            title='CSV 파일 선택', filetypes=[('CSV files', '*.csv')])
        if not path:
            return
        try:
            df = pd.read_csv(path, low_memory=False)
            missing = [c for c in FEATURE_COLS if c not in df.columns]
            if missing:
                messagebox.showerror('오류', f'필요한 컬럼 없음:\n{missing}')
                return
            self.csv_df = df
            messagebox.showinfo('완료',
                f'{len(df):,}개 투구 로드 완료!\n"투구 선택"으로 골라보세요.')
        except Exception as e:
            messagebox.showerror('오류', str(e))

    def _open_select(self):
        if self.csv_df is None:
            messagebox.showwarning('경고', 'CSV를 먼저 불러오세요.')
            return
        dialog = CSVSelectDialog(self.root, self.csv_df)
        self.root.wait_window(dialog.win)
        if dialog.result is not None:
            row = self.csv_df.loc[dialog.result]
            for col in FEATURE_COLS:
                self.entries1[col].delete(0, tk.END)
                self.entries1[col].insert(0, f'{row[col]:.4f}')
            if 'p_throws' in self.csv_df.columns:
                self.p_throws = str(row.get('p_throws', 'R'))
            self._predict_numeric()

    def _reset(self):
        for col in FEATURE_COLS:
            self.entries1[col].delete(0, tk.END)
            self.entries1[col].insert(0, SAMPLE_VALUES[col])
        self.result_label1.config(text='')
        self._draw_views(self.ax1_side, self.ax1_top, self.ax1_zone,
                         self.canvas1, None, None)
        self._draw_bar(self.ax_bar1, self.canvas_bar1, None, None)

    def _reset_draw(self):
        for col, _, _, _, _, _ in SLIDERS:
            self.slider_vars[col].set(float(SAMPLE_VALUES.get(col, 0)))
        self.result_label2.config(text='')
        self._sync_and_predict()


if __name__ == '__main__':
    root = tk.Tk()
    app  = PitchClassifierApp(root)
    root.mainloop()
