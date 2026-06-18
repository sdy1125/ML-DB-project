const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  })

  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const upstream = payload.details
    let upstreamMessage = ''
    if (typeof upstream === 'string') {
      try {
        upstreamMessage = JSON.parse(upstream).detail || upstream
      } catch {
        upstreamMessage = upstream
      }
    }
    throw new Error(upstreamMessage || payload.message || 'Yêu cầu thất bại.')
  }
  return payload
}

export const analysisApi = {
  health: () => request('/api/v1/health'),
  model: () => request('/api/v1/model'),
  explain: (payload) =>
    request('/api/v1/explanations', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  ask: (payload) =>
    request('/api/v1/assistant/questions', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
}
