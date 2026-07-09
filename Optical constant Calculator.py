import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def parse_concentration(col_name, prefix):
    """
    Extract concentration X from columns like CD_1, CDS_0.5, E_10uM.
    This version expects the part after prefix_ to begin with a number.
    """
    pattern = rf"^{prefix}_(.+)$"
    match = re.match(pattern, col_name)
    if not match:
        return None

    value = match.group(1)
    num_match = re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
    if not num_match:
        return None

    return float(num_match.group(0))


def get_grouped_columns(df, prefix):
    cols = []
    concs = []

    for col in df.columns:
        c = parse_concentration(str(col), prefix)
        if c is not None:
            cols.append(col)
            concs.append(c)

    order = np.argsort(concs)
    return [cols[i] for i in order], np.array([concs[i] for i in order], dtype=float)


def slope_vs_concentration(values, concentrations):
    """
    values shape: wavelength x concentration
    returns slope at each wavelength
    """
    slopes = []
    for row in values:
        mask = np.isfinite(row) & np.isfinite(concentrations)
        if np.sum(mask) < 2:
            slopes.append(np.nan)
        else:
            slopes.append(np.polyfit(concentrations[mask], row[mask], 1)[0])
    return np.array(slopes)


class OpticalConstantSolver:
    def __init__(self, root):
        self.root = root
        self.root.title("Optical Constant Solver")
        self.root.geometry("1250x800")

        self.df = None
        self.results = None
        self.file_path = None

        self.create_widgets()

    def create_widgets(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Button(top, text="Load Excel File", command=self.load_excel).pack(side="left", padx=5)

        ttk.Label(top, text="K:").pack(side="left", padx=(20, 5))
        self.k_entry = ttk.Entry(top, width=12)
        self.k_entry.insert(0, "39578")
        self.k_entry.pack(side="left")

        ttk.Label(top, text="l / cm:").pack(side="left", padx=(20, 5))
        self.l_entry = ttk.Entry(top, width=12)
        self.l_entry.insert(0, "1.0")
        self.l_entry.pack(side="left")

        ttk.Button(top, text="Calculate", command=self.calculate).pack(side="left", padx=20)
        ttk.Button(top, text="Save Results to Excel", command=self.save_excel).pack(side="left", padx=5)

        self.status = ttk.Label(self.root, text="Load an Excel file to begin.")
        self.status.pack(fill="x", padx=10)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

    def load_excel(self):
        path = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if not path:
            return

        try:
            self.df = pd.read_excel(path)
            self.file_path = path
            self.status.config(text=f"Loaded: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not read Excel file:\n{e}")

    def calculate(self):
        if self.df is None:
            messagebox.showwarning("Missing file", "Please load an Excel file first.")
            return

        try:
            K = float(self.k_entry.get())
            l = float(self.l_entry.get())
        except ValueError:
            messagebox.showerror("Input error", "K and l must be numeric.")
            return

        df = self.df.copy()
        wavelength = df.iloc[:, 0].to_numpy(dtype=float)

        CD_cols, conc_CD = get_grouped_columns(df, "CD")
        CDS_cols, conc_CDS = get_grouped_columns(df, "CDS")
        E_cols, conc_E = get_grouped_columns(df, "E")
        A_cols, conc_A = get_grouped_columns(df, "A")
        S_cols, conc_S = get_grouped_columns(df, "S")

        if not (len(CD_cols) and len(CDS_cols) and len(E_cols) and len(A_cols) and len(S_cols)):
            messagebox.showerror(
                "Column error",
                "Required columns were not found.\n\nExpected names such as:\n"
                "CD_1, CDS_1, E_1, A_1, S_1"
            )
            return

        if not (
            np.array_equal(conc_CD, conc_CDS)
            and np.array_equal(conc_CD, conc_E)
            and np.array_equal(conc_CD, conc_A)
            and np.array_equal(conc_CD, conc_S)
        ):
            messagebox.showerror(
                "Concentration mismatch",
                "The concentration labels after CD_, CDS_, E_, A_, and S_ must match exactly."
            )
            return

        concentrations = conc_CD

        CD = df[CD_cols].to_numpy(dtype=float)
        CDS = df[CDS_cols].to_numpy(dtype=float)
        E = df[E_cols].to_numpy(dtype=float)
        A = df[A_cols].to_numpy(dtype=float)
        S = df[S_cols].to_numpy(dtype=float)

        denominator_term = np.power(10, S) - 1
        denominator_term[np.isclose(denominator_term, 0)] = np.nan

        delta_A = (CDS + CD / denominator_term) / (K * (l + 1 / denominator_term))
        delta_S = CD / K - delta_A
        delta_E = delta_A + delta_S

        eps_E = slope_vs_concentration(E, concentrations)
        eps_A = slope_vs_concentration(A, concentrations)
        eps_S = slope_vs_concentration(S, concentrations)

        eps_dE = slope_vs_concentration(delta_E, concentrations)
        eps_dA = slope_vs_concentration(delta_A, concentrations)
        eps_dS = slope_vs_concentration(delta_S, concentrations)

        gE = eps_dE / eps_E
        gA = eps_dA / eps_A
        gS = eps_dS / eps_S

        self.results = {
            "wavelength": wavelength,
            "concentrations": concentrations,
            "CD": CD,
            "CDS": CDS,
            "delta_A": delta_A,
            "delta_S": delta_S,
            "delta_E": delta_E,
            "eps_E": eps_E,
            "eps_A": eps_A,
            "eps_S": eps_S,
            "eps_dE": eps_dE,
            "eps_dA": eps_dA,
            "eps_dS": eps_dS,
            "gE": gE,
            "gA": gA,
            "gS": gS,
        }

        self.status.config(text="Calculation completed.")
        self.plot_results()

    def clear_tabs(self):
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)

    def add_plot_tab(self, title, y_data, y_label, multi=False):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)

        fig = Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)

        wl = self.results["wavelength"]

        if multi:
            concentrations = self.results["concentrations"]
            for i, c in enumerate(concentrations):
                ax.plot(wl, y_data[:, i], label=f"{c:g}")
            ax.legend(title="Concentration", fontsize=8)
        else:
            ax.plot(wl, y_data)

        ax.set_xlabel("Wavelength")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_results(self):
        self.clear_tabs()
        r = self.results

        self.add_plot_tab("CD", r["CD"], "CD", multi=True)
        self.add_plot_tab("CDS", r["CDS"], "CDS", multi=True)
        self.add_plot_tab("ΔA", r["delta_A"], "ΔA", multi=True)
        self.add_plot_tab("ΔS", r["delta_S"], "ΔS", multi=True)
        self.add_plot_tab("ΔE", r["delta_E"], "ΔE", multi=True)

        self.add_plot_tab("εE", r["eps_E"], "εE")
        self.add_plot_tab("εA", r["eps_A"], "εA")
        self.add_plot_tab("εS", r["eps_S"], "εS")
        self.add_plot_tab("εΔE", r["eps_dE"], "εΔE")
        self.add_plot_tab("εΔA", r["eps_dA"], "εΔA")
        self.add_plot_tab("εΔS", r["eps_dS"], "εΔS")

        self.add_plot_tab("gE", r["gE"], "gE")
        self.add_plot_tab("gA", r["gA"], "gA")
        self.add_plot_tab("gS", r["gS"], "gS")

    def save_excel(self):
        if self.results is None:
            messagebox.showwarning("No results", "Please calculate results first.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if not path:
            return

        r = self.results
        wl = r["wavelength"]
        concs = r["concentrations"]

        spectra = pd.DataFrame({"Wavelength": wl})
        for i, c in enumerate(concs):
            label = f"{c:g}"
            spectra[f"CD_{label}"] = r["CD"][:, i]
            spectra[f"CDS_{label}"] = r["CDS"][:, i]
            spectra[f"deltaA_{label}"] = r["delta_A"][:, i]
            spectra[f"deltaS_{label}"] = r["delta_S"][:, i]
            spectra[f"deltaE_{label}"] = r["delta_E"][:, i]

        molar = pd.DataFrame({
            "Wavelength": wl,
            "epsilon_E": r["eps_E"],
            "epsilon_A": r["eps_A"],
            "epsilon_S": r["eps_S"],
            "epsilon_deltaE": r["eps_dE"],
            "epsilon_deltaA": r["eps_dA"],
            "epsilon_deltaS": r["eps_dS"],
        })

        gfactor = pd.DataFrame({
            "Wavelength": wl,
            "gE": r["gE"],
            "gA": r["gA"],
            "gS": r["gS"],
        })

        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                spectra.to_excel(writer, sheet_name="spectra", index=False)
                molar.to_excel(writer, sheet_name="molar coefficient", index=False)
                gfactor.to_excel(writer, sheet_name="dissymmetry factor", index=False)

            messagebox.showinfo("Saved", f"Results saved to:\n{path}")

        except Exception as e:
            messagebox.showerror("Save error", f"Could not save Excel file:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = OpticalConstantSolver(root)
    root.mainloop()