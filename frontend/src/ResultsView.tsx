import type { PredictionResult } from "./App";

const CLASS_COLORS: Record<string, string> = {
  metal: "bg-gray-500",
  semiconductor: "bg-amber-500",
  insulator: "bg-blue-500",
};

interface Props {
  result: PredictionResult;
}

export default function ResultsView({ result }: Props) {
  const colorClass = CLASS_COLORS[result.classification] ?? "bg-gray-400";

  return (
    <div className="mt-8 space-y-6">
      <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Predicted Band Gap
            </p>
            <p className="text-4xl font-bold tabular-nums">
              {result.predicted_band_gap.toFixed(3)}{" "}
              <span className="text-lg font-normal text-gray-500">eV</span>
            </p>
          </div>
          <span
            className={`${colorClass} text-white text-sm font-medium px-3 py-1 rounded-full`}
          >
            {result.classification}
          </span>
        </div>

        <div className="mt-4">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Uncertainty (1 std)
          </p>
          <div className="flex items-center gap-3 mt-1">
            <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full"
                style={{
                  width: `${Math.min(100, (1 - result.uncertainty / Math.max(result.predicted_band_gap, 0.1)) * 100)}%`,
                }}
              />
            </div>
            <span className="text-sm font-mono">
              ±{result.uncertainty.toFixed(3)} eV
            </span>
          </div>
        </div>
      </div>

      {result.similar_materials.length > 0 && (
        <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm">
          <h3 className="text-lg font-semibold mb-3">Similar Materials</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-500 dark:text-gray-400 border-b dark:border-gray-700">
                <th className="pb-2">ID</th>
                <th className="pb-2">Formula</th>
                <th className="pb-2 text-right">Band Gap (eV)</th>
              </tr>
            </thead>
            <tbody>
              {result.similar_materials.map((mat) => (
                <tr
                  key={mat.material_id}
                  className="border-b dark:border-gray-700 last:border-0"
                >
                  <td className="py-2 font-mono text-xs">{mat.material_id}</td>
                  <td className="py-2">{mat.formula}</td>
                  <td className="py-2 text-right tabular-nums">
                    {mat.band_gap.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
