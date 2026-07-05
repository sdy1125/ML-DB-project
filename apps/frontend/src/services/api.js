const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

function readableError(value) {
  if (!value) return ''
  if (typeof value === 'string') {
    try {
      return readableError(JSON.parse(value))
    } catch {
      return value
    }
  }
  if (Array.isArray(value)) {
    return value
      .map((item) => item?.msg || item?.message || readableError(item))
      .filter(Boolean)
      .join('; ')
  }
  if (typeof value === 'object') {
    if (value.detail) return readableError(value.detail)
    if (value.message) return readableError(value.message)
    if (value.error) return readableError(value.error)
    return JSON.stringify(value)
  }
  return String(value)
}

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
    const upstreamMessage = readableError(payload.details)
    const message = readableError(payload.message)
    throw new Error(upstreamMessage || message || 'Yêu cầu thất bại.')
  }
  return payload
}

export const analysisApi = {
  health: () => request('/api/v1/health'),
  model: () => request('/ml/model/info'),
  explain: (payload) =>
    request('/ml/explain', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  ask: (payload) =>
    request('/ml/ask', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  finalInsight: ({ country = 'Vietnam', year = '', useLlm = true } = {}) => {
    const params = new URLSearchParams({
      country,
      use_llm: String(useLlm),
    })
    if (year) {
      params.set('year', String(year))
    }
    return request(`/ml/insights/final?${params.toString()}`)
  },
}
