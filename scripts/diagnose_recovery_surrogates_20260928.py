"""CPU mathematical counterexamples, not biological experiment results."""
import json
from pathlib import Path
import numpy as np

OUT = Path('/mnt/huawei_deepcad/dinov3/outputs/00_reports/deepcad_method_20260927/theory_20260928')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d, v = 128, 0.01
    isotropic = v * np.eye(d)
    coherent = v * np.ones((d, d))
    assert np.allclose(np.diag(isotropic), np.diag(coherent))
    spectra = [np.linalg.eigvalsh(x) for x in (isotropic, coherent)]
    w = np.ones(d) / np.sqrt(d)
    risk = [float(w @ x @ w) for x in (isotropic, coherent)]
    assert np.isclose(risk[1] / risk[0], d)

    old = np.array([[2., 1.], [1., 0.], [0., 1.]])
    transform = np.diag([0.1, 1.])
    current = old @ transform
    recovered = current @ np.linalg.inv(transform)
    assert np.allclose(old, recovered)
    def similarities(x):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        return (x[1:] @ x[0]).tolist()
    before, after = similarities(old), similarities(current)
    assert before[0] > before[1] and after[0] < after[1]

    # Stationary noisy proxy; the underlying recovery quality never changes.
    rng = np.random.default_rng(0)
    observation = 0.30 + rng.normal(0, 0.08, 10000)
    monitor = 0.; floor = 0.; gate = 0.; gates = []
    for step, error in enumerate(observation):
        monitor = 0.9 * monitor + 0.1 * error
        if step < 32:
            floor = monitor
        else:
            floor = min(floor, monitor)
            gate = np.clip(gate + 0.05 * (monitor - floor - 0.02), 0, 1)
        gates.append(float(gate))
    result = dict(
        scope='Analytic counterexamples and stationary-noise simulation only; no biological validation.',
        same_channel_errors=dict(dimension=d,diagonal_mse=v,
            mean_mse=[float(np.trace(x)/d) for x in (isotropic,coherent)],
            worst_direction_mse=[float(x[-1]) for x in spectra],
            selected_unit_head_mse=risk,risk_ratio=risk[1]/risk[0]),
        exact_recovery_changed_retrieval=dict(old_cosines=before,current_cosines=after,
            max_reconstruction_error=float(abs(recovered-old).max()),decoder_operator_norm=10.),
        stationary_noise=dict(seed=0,steps=len(gates),true_error_mean=0.30,observation_std=0.08,
            final_floor=float(floor),final_gate=float(gate),last1000_gate_mean=float(np.mean(gates[-1000:]))))
    (OUT/'counterexamples.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    xx=np.arange(2)
    axes[0].bar(xx-.18,[v,v],.36,label='Mean/channel error',color='#659dbd')
    axes[0].bar(xx+.18,[float(x[-1]) for x in spectra],.36,label='Worst-direction error',color='#dd8452')
    axes[0].set_xticks(xx,['Independent errors','Correlated errors'])
    axes[0].set_ylabel('Expected squared score error')
    axes[0].set_title('Same channel errors, 128x worst risk')
    axes[0].legend(fontsize=8)
    axes[1].plot(gates,color='#985b9c')
    axes[1].set(xlabel='Monitor updates',ylabel='Adaptive gate',ylim=(-.02,1.05),
                title='Stationary quality + running-min floor')
    fig.suptitle('Synthetic counterexamples — not biological results',fontsize=12)
    fig.savefig(OUT/'surrogate_counterexamples.png',dpi=180)
    fig.savefig(OUT/'surrogate_counterexamples.pdf')
    plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
