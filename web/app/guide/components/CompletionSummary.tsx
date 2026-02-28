"use client";

import { useState } from "react";
import { CheckCircle2, RotateCcw, History, ChevronDown, ChevronUp } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { useTranslation } from "react-i18next";
import { processLatexContent } from "@/lib/latex";
import { LearningHistory } from "../types";

interface CompletionSummaryProps {
  summary: string;
  learningHistory: LearningHistory[];
  onRestart: () => void;
}

export default function CompletionSummary({ summary, learningHistory, onRestart }: CompletionSummaryProps) {
  const { t } = useTranslation();
  const [selectedHistory, setSelectedHistory] = useState<LearningHistory | null>(null);
  const [showHistory, setShowHistory] = useState(false);

  const tableComponents = {
    table: ({ node, ...props }: any) => (
      <div className="overflow-x-auto my-6 rounded-lg border border-slate-200 shadow-sm">
        <table className="min-w-full divide-y divide-slate-200 text-sm" {...props} />
      </div>
    ),
    thead: ({ node, ...props }: any) => <thead className="bg-slate-50" {...props} />,
    th: ({ node, ...props }: any) => (
      <th className="px-4 py-3 text-left font-semibold text-slate-700 whitespace-nowrap border-b border-slate-200" {...props} />
    ),
    tbody: ({ node, ...props }: any) => <tbody className="divide-y divide-slate-100 bg-white" {...props} />,
    td: ({ node, ...props }: any) => (
      <td className="px-4 py-3 text-slate-600 border-b border-slate-100" {...props} />
    ),
    tr: ({ node, ...props }: any) => (
      <tr className="hover:bg-slate-50/50 transition-colors" {...props} />
    ),
  };

  const displaySummary = selectedHistory ? selectedHistory.summary : summary;
  const displayTitle = selectedHistory
    ? selectedHistory.notebook_name
    : t("Learning Summary");

  return (
    <div className="flex-1 bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 flex flex-col overflow-hidden relative">
      {/* Header */}
      <div className="p-4 border-b border-slate-100 dark:border-slate-700 bg-gradient-to-r from-emerald-50 to-indigo-50 dark:from-emerald-900/20 dark:to-indigo-900/20 shrink-0">
        <div className="flex items-center justify-between">
          <h2 className="font-bold text-slate-800 dark:text-slate-100 flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
            {displayTitle}
          </h2>
          <div className="flex items-center gap-2">
            {learningHistory.length > 0 && (
              <button
                onClick={() => setShowHistory(!showHistory)}
                className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-slate-600 dark:text-slate-300 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-600 transition-colors"
              >
                <History className="w-3.5 h-3.5" />
                {t("History")} ({learningHistory.length})
                {showHistory ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              </button>
            )}
            <button
              onClick={onRestart}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-slate-600 dark:text-slate-300 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-600 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              {t("New Learning")}
            </button>
          </div>
        </div>

        {/* History dropdown */}
        {showHistory && (
          <div className="mt-3 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg overflow-hidden">
            <button
              onClick={() => { setSelectedHistory(null); setShowHistory(false); }}
              className={`w-full text-left px-3 py-2 text-sm hover:bg-slate-50 dark:hover:bg-slate-600 transition-colors border-b border-slate-100 dark:border-slate-600 ${!selectedHistory ? "text-indigo-600 font-medium" : "text-slate-700 dark:text-slate-300"}`}
            >
              {t("Current Session")}
            </button>
            {learningHistory.map((h) => (
              <button
                key={h.id}
                onClick={() => { setSelectedHistory(h); setShowHistory(false); }}
                className={`w-full text-left px-3 py-2 text-sm hover:bg-slate-50 dark:hover:bg-slate-600 transition-colors border-b border-slate-100 dark:border-slate-600 last:border-0 ${selectedHistory?.id === h.id ? "text-indigo-600 font-medium" : "text-slate-700 dark:text-slate-300"}`}
              >
                <div className="font-medium truncate">{h.notebook_name}</div>
                <div className="text-xs text-slate-400 mt-0.5">
                  {new Date(h.completed_at).toLocaleDateString()} · {h.knowledge_points.length} {t("knowledge points")}
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Summary Content */}
      <div className="flex-1 overflow-y-auto p-8 bg-white dark:bg-slate-800">
        <div className="prose prose-slate dark:prose-invert prose-headings:font-bold prose-h1:text-2xl prose-h2:text-xl max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
            components={tableComponents}
          >
            {processLatexContent(displaySummary || "")}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
