"""Utility functions for subliminal learning experiments."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import sys

# matplotlibの設定
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def get_token_id(tokenizer, text: str) -> int:
    """
    特別トークンを除外してエンコードし、最後のサブトークンIDを返す。
    単一トークンでない場合でも頑健に動く（最後のサブトークンを使用）。
    """
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"Text produced no tokens: {text!r}")
    return ids[-1]

def is_english_num(s):
    """Check if string is an English number."""
    return s.isdecimal() and s.isdigit() and s.isascii()

def ensure_output_dir() -> Path:
    """出力ディレクトリを確保"""
    try:
        base_path = Path(__file__).resolve().parent
    except NameError:
        base_path = Path.cwd()
    out = base_path / "outputs"
    out.mkdir(exist_ok=True)
    return out

def save_dataframe_as_png(df, filename: str, title: str = None):
    """DataFrameを表としてPNG保存"""
    out_dir = ensure_output_dir()
    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 2), max(4, len(df) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    if title:
        fig.suptitle(title, fontsize=12, y=0.95)
    
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_dir / filename}")

def save_plotly_as_png(fig, filename: str):
    """Plotly図をPNG保存（kaleidoが必要）"""
    out_dir = ensure_output_dir()
    
    # Try different methods to save plotly figure
    saved = False
    
    # Method 1: Try kaleido
    try:
        import kaleido
        fig.write_image(str(out_dir / filename), scale=2)
        print(f"Saved with kaleido: {out_dir / filename}")
        saved = True
    except ImportError:
        print("kaleido not installed, trying alternative methods...")
    except Exception as e:
        print(f"kaleido failed: {e}")
    
    # Method 2: Try plotly-orca
    if not saved:
        try:
            fig.write_image(str(out_dir / filename), engine="orca", scale=2)
            print(f"Saved with orca: {out_dir / filename}")
            saved = True
        except Exception as e:
            print(f"orca failed: {e}")
    
    # Method 3: Save as HTML as fallback
    if not saved:
        try:
            html_file = str(out_dir / filename.replace('.png', '.html'))
            fig.write_html(html_file)
            print(f"Warning: Could not save as PNG. Saved as HTML instead: {html_file}")
            print("To save as PNG, install kaleido: pip install kaleido")
        except Exception as e:
            print(f"Error: Could not save plotly figure at all: {e}")
    
    return saved

def save_matplotlib_as_png(fig, filename: str):
    """matplotlib図をPNG保存"""
    out_dir = ensure_output_dir()
    fig.savefig(out_dir / filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_dir / filename}")
