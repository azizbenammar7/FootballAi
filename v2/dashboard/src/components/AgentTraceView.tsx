import { ListChecks } from 'lucide-react'
import type { AgentToolCall, AgentTrace } from '../types'

const STOP_LABEL: Record<string, string> = {
  answered: 'Answered',
  unsupported_question: 'Question outside FootballAI',
  evidence_not_available: 'Evidence not available',
  no_relevant_claim: 'No relevant claim selected',
  selection_rejected: 'Selection rejected',
  citation_rejected: 'Citation rejected',
  cross_run_rejected: 'Cross-run reference rejected',
  tool_budget_exhausted: 'Tool limit reached',
  turn_budget_exhausted: 'Turn limit reached',
  deadline_exceeded: 'Time limit reached',
  prompt_budget_exceeded: 'Prompt size limit reached',
  model_timeout: 'Adapter timed out',
  model_refusal: 'Adapter refused',
  model_truncated: 'Adapter reply truncated',
  malformed_response: 'Adapter reply malformed',
  provider_error: 'Adapter unavailable',
}

const STATUS_LABEL: Record<string, string> = {
  ok: 'Executed — evidence returned',
  no_evidence: 'Executed — nothing for this question',
  failed: 'Executed — result not usable',
  rejected: 'Rejected — never ran',
}

function plural(count: number, one: string, many: string) {
  return `${count} ${count === 1 ? one : many}`
}

function ToolCallRow({ call }: { call: AgentToolCall }) {
  return (
    <li className={`agent-trace__tool agent-trace__tool--${call.status}`}>
      <div>
        {call.tool ? <code>{call.tool}</code> : <strong>Outside the tool registry</strong>}
        <span className="agent-trace__status">{STATUS_LABEL[call.status] ?? call.status}</span>
      </div>
      <small>Turn {call.turn} · {call.duration_ms.toFixed(2)} ms{call.error_code ? ` · ${call.error_code}` : ''}</small>
      {call.claim_ids_returned.length > 0 && (
        <p className="agent-trace__claims">{call.claim_ids_returned.map((id) => <code key={id}>{id}</code>)}</p>
      )}
    </li>
  )
}

// Every line comes from the server's trace. Nothing here is inferred or animated,
// so a step that did not happen cannot appear.
export function AgentTraceView({ trace }: { trace: AgentTrace }) {
  const executed = trace.tool_calls.filter((call) => call.status !== 'rejected').length
  const stopLabel = STOP_LABEL[trace.termination] ?? trace.termination
  return (
    <details className="agent-trace">
      <summary><ListChecks /> Tool execution trace · {executed} of {trace.tool_call_limit} tool calls executed · {stopLabel}</summary>
      <dl className="agent-trace__facts">
        <div><dt>Ended</dt><dd>{stopLabel}</dd></div>
        <div><dt>Adapter</dt><dd>{trace.model}</dd></div>
        <div><dt>Turns</dt><dd>{trace.model_turns} of {trace.model_turn_limit}</dd></div>
        <div><dt>Time</dt><dd>{trace.duration_ms.toFixed(1)} ms</dd></div>
        <div><dt>Routing</dt><dd>{trace.routing_version}</dd></div>
        <div><dt>Correlation id</dt><dd>{trace.correlation_id}</dd></div>
        {trace.tool_error_code && <div><dt>Tool error</dt><dd>{trace.tool_error_code}</dd></div>}
      </dl>
      {trace.tool_calls.length > 0
        ? <ol className="agent-trace__tools">{trace.tool_calls.map((call) => <ToolCallRow key={call.sequence} call={call} />)}</ol>
        : <p className="agent-trace__note">{trace.model_turns === 0 ? 'No tool ran. The server stopped this question before any adapter or tool was called.' : 'The adapter called no tool.'}</p>}
      {trace.tool_calls_refused > 0 && <p className="agent-trace__note">{plural(trace.tool_calls_refused, 'requested call was', 'requested calls were')} refused before running.</p>}
      {trace.selected_ids_rejected > 0 && <p className="agent-trace__note">{plural(trace.selected_ids_rejected, 'selected id was', 'selected ids were')} rejected.</p>}
      {trace.claim_ids_rendered.length > 0 && (
        <div className="agent-trace__rendered">
          <h4>Claims shown in the answer</h4>
          <p className="agent-trace__claims">{trace.claim_ids_rendered.map((id) => <code key={id}>{id}</code>)}</p>
        </div>
      )}
    </details>
  )
}
