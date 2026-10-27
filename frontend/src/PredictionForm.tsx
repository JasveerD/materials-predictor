import { useState } from "react";

type InputMode = "cif" | "mpid";

interface Props {
  onPredict: (payload: { cif?: string; material_id?: string }) => void;
  loading: boolean;
}

export default function PredictionForm({ onPredict, loading }: Props) {
  const [mode, setMode] = useState<InputMode>("cif");
  const [cif, setCif] = useState("");
  const [mpid, setMpid] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (mode === "cif") {
      onPredict({ cif });
    } else {
      onPredict({ material_id: mpid });
    }
  };

  const canSubmit = mode === "cif" ? cif.trim().length > 0 : mpid.trim().length > 0;

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => setMode("cif")}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            mode === "cif"
              ? "bg-blue-600 text-white"
              : "bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300"
          }`}
        >
          Paste CIF
        </button>
        <button
          type="button"
          onClick={() => setMode("mpid")}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            mode === "mpid"
              ? "bg-blue-600 text-white"
              : "bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300"
          }`}
        >
          Materials Project ID
        </button>
      </div>

      {mode === "cif" ? (
        <textarea
          value={cif}
          onChange={(e) => setCif(e.target.value)}
          placeholder="Paste CIF file content here..."
          rows={10}
          className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-3 font-mono text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      ) : (
        <input
          type="text"
          value={mpid}
          onChange={(e) => setMpid(e.target.value)}
          placeholder="e.g. mp-149"
          className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      )}

      <button
        type="submit"
        disabled={!canSubmit || loading}
        className="px-6 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg font-medium transition-colors"
      >
        {loading ? "Predicting..." : "Predict Band Gap"}
      </button>
    </form>
  );
}
