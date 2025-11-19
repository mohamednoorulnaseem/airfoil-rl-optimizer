import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from stable_baselines3 import PPO

from airfoil_gen import naca4
from aero_eval import aero_score
from airfoil_env import AirfoilEnv


@st.cache_resource
def load_model_and_env():
    env = AirfoilEnv()
    model = PPO.load("models/ppo_airfoil_fake.zip")
    return model, env


def run_rl_optimization():
    model, env = load_model_and_env()
    obs, info = env.reset()
    best_params = env.params.copy()
    best_ld = -1.0

    for _ in range(env.max_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        ld = info["L/D"]
        if ld > best_ld:
            best_ld = ld
            best_params = env.params.copy()

        if terminated or truncated:
            break

    return best_params, best_ld


def plot_airfoil(m, p, t, title="Airfoil"):
    xu, yu, xl, yl = naca4(m, p, t)

    fig, ax = plt.subplots()
    ax.plot(xu, yu, label="Upper surface")
    ax.plot(xl, yl, label="Lower surface")
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def main():
    st.title("RL + XFOIL Airfoil Optimizer ✈️")

    st.sidebar.header("Manual Airfoil Parameters")

    # Sliders for NACA-like parameters
    m = st.sidebar.slider("Max camber m (0–0.06)", 0.0, 0.06, 0.02, 0.005)
    p = st.sidebar.slider("Camber position p (0.1–0.7)", 0.1, 0.7, 0.4, 0.05)
    t = st.sidebar.slider("Thickness t (0.08–0.18)", 0.08, 0.18, 0.12, 0.01)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Manual Airfoil")
        fig_manual = plot_airfoil(m, p, t, title=f"Manual: m={m:.3f}, p={p:.3f}, t={t:.3f}")
        st.pyplot(fig_manual)

        if st.button("Evaluate Manual Airfoil (XFOIL / fallback)"):
            Cl, Cd = aero_score(m, p, t)
            ld = Cl / (Cd + 1e-6)
            st.write(f"**Cl:** {Cl:.4f}")
            st.write(f"**Cd:** {Cd:.5f}")
            st.write(f"**L/D:** {ld:.2f}")

    with col2:
        st.subheader("RL-Optimized Airfoil")

        if st.button("Run RL Optimization"):
            with st.spinner("Optimizing with PPO agent..."):
                best_params, best_ld = run_rl_optimization()

            m_opt, p_opt, t_opt = best_params
            Cl_opt, Cd_opt = aero_score(m_opt, p_opt, t_opt)
            fig_opt = plot_airfoil(
                float(m_opt),
                float(p_opt),
                float(t_opt),
                title=f"Optimized: m={m_opt:.3f}, p={p_opt:.3f}, t={t_opt:.3f}",
            )
            st.pyplot(fig_opt)

            st.success("Optimization complete!")
            st.write(f"**Optimized m, p, t:** {m_opt:.4f}, {p_opt:.4f}, {t_opt:.4f}")
            st.write(f"**Cl (opt):** {Cl_opt:.4f}")
            st.write(f"**Cd (opt):** {Cd_opt:.5f}")
            st.write(f"**L/D (opt):** {best_ld:.2f}")

    st.markdown("---")
    st.markdown(
        "This tool uses **XFOIL** for aerodynamic evaluation and a **PPO agent** "
        "trained in a custom Gymnasium environment to optimize NACA-like airfoil shapes."
    )


if __name__ == "__main__":
    main()
