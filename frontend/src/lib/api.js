// Thin API client. Everything the dashboard shows comes from the Python
// service — no numbers are computed or invented in the browser.

const json = async (r) => {
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
  return r.json()
}

export const getHealth    = () => fetch('/api/health').then(json)
export const getScenarios = () => fetch('/api/scenarios').then(json)
export const getStatus    = () => fetch('/api/status').then(json)
export const getAlerts    = () => fetch('/api/alerts').then(json)
export const getHistory   = () => fetch('/api/history').then(json)

export const startSession = (scenario, speed, seed = 42) =>
  fetch('/api/session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario, speed, seed }),
  }).then(json)

export const control = (action, speed) =>
  fetch('/api/session/control', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, speed }),
  }).then(json)

export function wsUrl() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${location.host}/ws`
}
