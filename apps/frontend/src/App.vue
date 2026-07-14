<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { analysisApi } from './services/api'

const indicators = [
  {
    key: 'n_sdg16_cpi',
    label: 'Cảm nhận tham nhũng',
    short: 'CPI',
    description: 'Mức độ minh bạch và kiểm soát tham nhũng',
  },
  {
    key: 'n_sdg16_detain',
    label: 'Tạm giam chưa xét xử',
    short: 'Detain',
    description: 'Chất lượng thủ tục và bảo đảm tư pháp',
  },
  {
    key: 'n_sdg16_justice',
    label: 'Tiếp cận công lý',
    short: 'Justice',
    description: 'Khả năng tiếp cận thể chế pháp lý',
  },
  {
    key: 'n_sdg16_rsf',
    label: 'Tự do báo chí',
    short: 'RSF',
    description: 'Môi trường thông tin và trách nhiệm giải trình',
  },
  {
    key: 'n_sdg16_weaponsexp',
    label: 'Xuất khẩu vũ khí',
    short: 'Weapons',
    description: 'Chỉ số liên quan đến hòa bình và an ninh',
  },
]

const countries = ['Vietnam', 'Thailand', 'Malaysia', 'Indonesia', 'Singapore']
const form = reactive({
  country: 'Vietnam',
  year: 2025,
  features: Object.fromEntries(indicators.map((item) => [item.key, 50])),
})

const health = ref(null)
const model = ref(null)
const result = ref(null)
const answer = ref(null)
const finalInsight = ref(null)
const question = ref('Việt Nam nên ưu tiên cải thiện chỉ số nào và vì sao?')
const loadingAnalysis = ref(false)
const loadingAnswer = ref(false)
const loadingFinalInsight = ref(false)
const error = ref('')

const modelReady = computed(() => health.value?.model_ready === true)
const topContributions = computed(() => result.value?.contributions?.slice(0, 5) || [])
const maxContribution = computed(() => {
  const values = topContributions.value.map((item) => Math.abs(item.contribution))
  return Math.max(...values, 0.001)
})
const finalWeakestMax = computed(() => {
  const values = finalInsight.value?.weakest_indicators?.map((item) => Math.abs(item.contribution)) || []
  return Math.max(...values, 0.001)
})
const baseForecasts = computed(() => finalInsight.value?.gru_forecast_baseline || [])
const leakageTopCorrelations = computed(() => finalInsight.value?.leakage_report?.top_abs_correlations || [])
const panelTopVif = computed(() => finalInsight.value?.panel_ols_diagnostics?.top_vif || [])

function payload() {
  return {
    country: form.country,
    year: Number(form.year),
    features: Object.fromEntries(
      Object.entries(form.features).map(([key, value]) => [key, Number(value)]),
    ),
  }
}

function contributionWidth(value) {
  return `${Math.max((Math.abs(value) / maxContribution.value) * 100, 4)}%`
}

function finalContributionWidth(value) {
  return `${Math.max((Math.abs(value) / finalWeakestMax.value) * 100, 4)}%`
}

function displayFeature(key) {
  return indicators.find((item) => item.key === key)?.label || key
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'N/A'
  return Number(value).toFixed(digits)
}

async function loadStatus() {
  try {
    const [healthResponse, modelResponse] = await Promise.all([
      analysisApi.health(),
      analysisApi.model(),
    ])
    health.value = healthResponse
    model.value = modelResponse
  } catch (statusError) {
    error.value = statusError.message
  }
}

async function analyze() {
  loadingAnalysis.value = true
  error.value = ''
  try {
    result.value = await analysisApi.explain(payload())
  } catch (analysisError) {
    error.value = analysisError.message
  } finally {
    loadingAnalysis.value = false
  }
}

async function askAssistant() {
  loadingAnswer.value = true
  error.value = ''
  try {
    answer.value = await analysisApi.ask({
      question: question.value,
      ...payload(),
    })
  } catch (assistantError) {
    error.value = assistantError.message
  } finally {
    loadingAnswer.value = false
  }
}

async function generateFinalInsight() {
  loadingFinalInsight.value = true
  error.value = ''
  try {
    finalInsight.value = await analysisApi.finalInsight({
      country: form.country,
      year: Number(form.year),
      useLlm: true,
    })
  } catch (insightError) {
    error.value = insightError.message
  } finally {
    loadingFinalInsight.value = false
  }
}

onMounted(loadStatus)
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <a class="brand" href="#">
        <span class="brand-mark">16</span>
        <span>
          <strong>SDG Intelligence</strong>
          <small>Peace · Justice · Strong Institutions</small>
        </span>
      </a>
      <nav>
        <a href="#analysis">Phân tích</a>
        <a href="#assistant">Khuyến nghị</a>
        <a href="#method">Phương pháp</a>
      </nav>
      <div class="system-state" :class="{ ready: modelReady }">
        <span></span>
        {{ modelReady ? 'Model sẵn sàng' : 'Chờ model' }}
      </div>
    </header>

    <main>
      <section class="hero">
        <div class="hero-copy">
          <p class="eyebrow">DATA-DRIVEN POLICY LAB</p>
          <h1>Nhìn thấy điều gì đang<br /><em>kéo điểm Việt Nam xuống.</em></h1>
          <p class="hero-description">
            Nền tảng kết hợp Big Data, Machine Learning và AI giải thích để dự
            đoán Goal16, nhận diện chỉ số có ảnh hưởng lớn và hỗ trợ xây dựng
            ưu tiên chính sách có căn cứ.
          </p>
          <div class="hero-actions">
            <a class="primary-button" href="#analysis">Bắt đầu phân tích</a>
            <a class="text-link" href="#method">Xem phương pháp <span>→</span></a>
          </div>
        </div>
        <div class="hero-visual" aria-hidden="true">
          <div class="orb orb-one"></div>
          <div class="orb orb-two"></div>
          <div class="score-card">
            <div class="score-card-heading">
              <span>Dự báo Goal16</span>
              <span class="live-pill">MODEL</span>
            </div>
            <strong>{{ result?.predicted_score?.toFixed(2) || '—' }}</strong>
            <small>{{ form.country }} · {{ form.year }}</small>
            <div class="mini-lines">
              <i style="height: 34%"></i><i style="height: 46%"></i>
              <i style="height: 40%"></i><i style="height: 62%"></i>
              <i style="height: 58%"></i><i style="height: 78%"></i>
              <i class="accent" style="height: 91%"></i>
            </div>
          </div>
          <div class="floating-note">
            <span>↗</span>
            <div><strong>Explainable AI</strong><small>Contribution theo từng chỉ số</small></div>
          </div>
        </div>
      </section>

      <section id="analysis" class="workspace-section">
        <div class="section-heading">
          <div>
            <p class="eyebrow">COUNTRY ANALYSIS</p>
            <h2>Mô phỏng điểm số</h2>
          </div>
          <p>
            Điều chỉnh dữ liệu đầu vào để xem dự đoán và đóng góp của từng chỉ số.
            Kết quả là ước lượng mô hình, không phải điểm chính thức của SDSN.
          </p>
        </div>

        <div v-if="error" class="alert">
          <strong>Chưa thể hoàn tất yêu cầu</strong>
          <span>{{ error }}</span>
          <button type="button" @click="error = ''">×</button>
        </div>

        <div class="analysis-grid">
          <form class="control-panel" @submit.prevent="analyze">
            <div class="panel-header">
              <div>
                <span class="step-label">01 / INPUT</span>
                <h3>Dữ liệu quốc gia</h3>
              </div>
              <button class="reset-button" type="button" @click="indicators.forEach((i) => form.features[i.key] = 50)">
                Đặt lại
              </button>
            </div>

            <div class="form-row">
              <label>
                Quốc gia
                <select v-model="form.country">
                  <option v-for="country in countries" :key="country">{{ country }}</option>
                </select>
              </label>
              <label>
                Năm
                <input v-model.number="form.year" type="number" min="2000" max="2030" />
              </label>
            </div>

            <div class="indicator-list">
              <label v-for="indicator in indicators" :key="indicator.key" class="indicator">
                <span class="indicator-copy">
                  <b>{{ indicator.short }}</b>
                  <span>
                    <strong>{{ indicator.label }}</strong>
                    <small>{{ indicator.description }}</small>
                  </span>
                </span>
                <span class="range-control">
                  <input
                    v-model.number="form.features[indicator.key]"
                    type="range"
                    min="0"
                    max="100"
                    step="0.1"
                  />
                  <input
                    v-model.number="form.features[indicator.key]"
                    class="number-input"
                    type="number"
                    min="0"
                    max="100"
                    step="0.1"
                  />
                </span>
              </label>
            </div>

            <button class="analyze-button" type="submit" :disabled="loadingAnalysis">
              <span>{{ loadingAnalysis ? 'Đang chạy mô hình...' : 'Phân tích dữ liệu' }}</span>
              <span>→</span>
            </button>
          </form>

          <div class="result-panel">
            <div class="panel-header">
              <div>
                <span class="step-label">02 / OUTPUT</span>
                <h3>Kết quả giải thích</h3>
              </div>
              <span v-if="result" class="version-pill">{{ result.model_version }}</span>
            </div>

            <div v-if="!result" class="empty-result">
              <div class="empty-icon">⌁</div>
              <h3>Chưa có kết quả phân tích</h3>
              <p>
                Điền các chỉ số bên trái và chạy mô hình. Nếu model chưa được
                huấn luyện, hãy chạy Spark pipeline trước.
              </p>
            </div>

            <template v-else>
              <div class="score-summary">
                <div>
                  <span>Goal16 dự đoán</span>
                  <strong>{{ result.predicted_score.toFixed(2) }}</strong>
                </div>
                <div>
                  <span>Baseline</span>
                  <strong>{{ result.baseline.toFixed(2) }}</strong>
                </div>
                <div>
                  <span>Quốc gia / năm</span>
                  <strong>{{ result.country }} · {{ result.year }}</strong>
                </div>
              </div>

              <div class="contribution-header">
                <div>
                  <h4>Chỉ số tác động mạnh nhất</h4>
                  <p>Độ dài thể hiện độ lớn contribution sau chuẩn hóa.</p>
                </div>
                <span><i class="positive-dot"></i> Tích cực</span>
                <span><i class="negative-dot"></i> Tiêu cực</span>
              </div>

              <div class="contribution-list">
                <div v-for="item in topContributions" :key="item.feature" class="contribution-row">
                  <div class="contribution-label">
                    <strong>{{ displayFeature(item.feature) }}</strong>
                    <span>{{ item.contribution > 0 ? '+' : '' }}{{ item.contribution.toFixed(3) }}</span>
                  </div>
                  <div class="bar-track">
                    <span
                      :class="item.contribution >= 0 ? 'bar-positive' : 'bar-negative'"
                      :style="{ width: contributionWidth(item.contribution) }"
                    ></span>
                  </div>
                </div>
              </div>
              <p v-if="result.warning" class="result-warning">{{ result.warning }}</p>
            </template>
          </div>
        </div>
      </section>

      <section id="assistant" class="assistant-section">
        <div class="assistant-intro">
          <p class="eyebrow light">POLICY RECOMMENDATION</p>
          <h2>Đặt câu hỏi cho trợ lý chính sách.</h2>
          <p>
            Trợ lý kết hợp kết quả mô hình với tài liệu được truy xuất từ kho
            tri thức. LLM chỉ diễn giải, không tự tính điểm số.
          </p>
          <div class="assistant-meta">
            <span>RAG {{ health?.rag_enabled ? 'đã bật' : 'chưa cấu hình' }}</span>
            <span>LLM {{ health?.llm_enabled ? 'đã bật' : 'chưa cấu hình' }}</span>
          </div>
        </div>
        <div class="chat-card">
          <label for="question">Câu hỏi của bạn</label>
          <textarea id="question" v-model="question" rows="4"></textarea>
          <button type="button" :disabled="loadingAnswer" @click="askAssistant">
            {{ loadingAnswer ? 'Đang tổng hợp...' : 'Tạo khuyến nghị' }}
            <span>✦</span>
          </button>
          <div v-if="answer" class="answer-box">
            <span class="answer-label">TRỢ LÝ SDG16</span>
            <p>{{ answer.answer }}</p>
            <small v-if="answer.provider">Provider: {{ answer.provider }}</small>
          </div>
        </div>
      </section>

      <section class="workspace-section">
        <div class="section-heading">
          <div>
            <p class="eyebrow">FINAL OUTPUT</p>
            <h2>Đích cuối: ML + Explainability + RAG + LLM</h2>
          </div>
          <p>
            Hệ thống tự lấy metadata mô hình, xác định chỉ số kéo điểm Việt Nam xuống,
            truy xuất tài liệu RAG và sinh khuyến nghị chính sách cuối.
          </p>
        </div>

        <div class="result-panel">
          <div class="panel-header">
            <div>
              <span class="step-label">06 / RAG + LLM</span>
              <h3>Báo cáo tổng hợp cuối</h3>
            </div>
            <button class="analyze-button" type="button" :disabled="loadingFinalInsight" @click="generateFinalInsight">
              {{ loadingFinalInsight ? 'Đang sinh báo cáo...' : 'Sinh output cuối' }}
            </button>
          </div>

          <div v-if="!finalInsight" class="empty-result">
            <div class="empty-icon">✦</div>
            <h3>Chưa sinh output cuối</h3>
            <p>Bấm nút để tạo báo cáo gồm điểm VN hiện tại, chỉ số yếu nhất, trạng thái drill-down tỉnh, kịch bản và khuyến nghị.</p>
          </div>

          <template v-else>
            <div class="score-summary">
              <div>
                <span>Điểm VN hiện tại</span>
                <strong>{{ finalInsight.current_score.toFixed(2) }}</strong>
              </div>
              <div>
                <span>Điểm quan sát</span>
                <strong>{{ finalInsight.observed_score?.toFixed(2) || 'N/A' }}</strong>
              </div>
              <div>
                <span>Model</span>
                <strong>{{ finalInsight.model_version }}</strong>
              </div>
            </div>

            <div class="diagnostic-grid">
              <article v-if="finalInsight.panel_ols_diagnostics" class="diagnostic-card">
                <span class="answer-label">PANEL OLS DIAGNOSTICS</span>
                <div class="metric-pairs">
                  <p><b>R² overall</b><strong>{{ formatNumber(finalInsight.panel_ols_diagnostics.r2_overall, 4) }}</strong></p>
                  <p><b>RMSE</b><strong>{{ formatNumber(finalInsight.panel_ols_diagnostics.rmse, 4) }}</strong></p>
                  <p><b>MAE</b><strong>{{ formatNumber(finalInsight.panel_ols_diagnostics.mae, 4) }}</strong></p>
                  <p><b>DW mean</b><strong>{{ formatNumber(finalInsight.panel_ols_diagnostics.durbin_watson_panel_mean, 4) }}</strong></p>
                </div>
                <small>
                  Heteroskedasticity flag:
                  {{ finalInsight.panel_ols_diagnostics.heteroskedasticity_flag ? 'Có cảnh báo' : 'Không bật cờ' }}
                </small>
                <div v-if="panelTopVif.length" class="mini-table">
                  <div v-for="item in panelTopVif" :key="item.feature">
                    <span>{{ item.feature }}</span>
                    <strong>{{ formatNumber(item.vif, 2) }}</strong>
                  </div>
                </div>
              </article>

              <article v-if="finalInsight.leakage_report" class="diagnostic-card">
                <span class="answer-label">XGBOOST LEAKAGE CHECK</span>
                <div class="status-pill clean">
                  {{ finalInsight.leakage_report.leakage_status }}
                </div>
                <p class="diagnostic-note">
                  Exact duplicate: {{ finalInsight.leakage_report.exact_duplicate_features?.length || 0 }};
                  high corr ≥ 0.98:
                  {{ Object.keys(finalInsight.leakage_report.high_corr_features_abs_ge_0_98 || {}).length }}
                </p>
                <p v-if="finalInsight.leakage_report.circular_target_warning" class="diagnostic-note warning">
                  {{ finalInsight.leakage_report.circular_target_warning }}
                </p>
                <div v-if="leakageTopCorrelations.length" class="mini-table">
                  <div v-for="item in leakageTopCorrelations" :key="item.feature">
                    <span>{{ item.feature }}</span>
                    <strong>{{ formatNumber(item.abs_correlation_with_goal16, 3) }}</strong>
                  </div>
                </div>
              </article>
            </div>

            <div class="contribution-header">
              <div>
                <h4>Chỉ số kéo điểm xuống mạnh nhất</h4>
                <p>{{ finalInsight.explainability_source }}</p>
              </div>
            </div>
            <div class="contribution-list">
              <div v-for="item in finalInsight.weakest_indicators.slice(0, 5)" :key="item.feature" class="contribution-row">
                <div class="contribution-label">
                  <strong>{{ item.label }}</strong>
                  <span>{{ item.contribution.toFixed(3) }}</span>
                </div>
                <div class="bar-track">
                  <span class="bar-negative" :style="{ width: finalContributionWidth(item.contribution) }"></span>
                </div>
              </div>
            </div>

            <div class="method-grid">
              <article v-for="scenario in finalInsight.forecasts" :key="`${scenario.scenario}-${scenario.year}`">
                <span>{{ scenario.scenario }} · {{ scenario.year }}</span>
                <h3>{{ scenario.predicted_score.toFixed(2) }}</h3>
                <p>Δ so với hiện tại: {{ scenario.delta_vs_current > 0 ? '+' : '' }}{{ scenario.delta_vs_current.toFixed(2) }}</p>
              </article>
            </div>

            <div v-if="baseForecasts.length" class="forecast-card">
              <div class="contribution-header compact">
                <div>
                  <h4>GRU forecast baseline mới</h4>
                  <p>Recursive forecast dùng trend feature Việt Nam có damping, không còn phẳng.</p>
                </div>
              </div>
              <div class="forecast-strip">
                <article v-for="row in baseForecasts" :key="row.year">
                  <span>{{ row.year }}</span>
                  <strong>{{ formatNumber(row.predicted_goal16, 2) }}</strong>
                  <small>trend {{ formatNumber(row.feature_trend_norm, 3) }}</small>
                </article>
              </div>
            </div>

            <div class="answer-box">
              <span class="answer-label">KHUYẾN NGHỊ CUỐI</span>
              <p>{{ finalInsight.recommendation }}</p>
              <small>{{ finalInsight.province.message }}</small>
            </div>
          </template>
        </div>
      </section>

      <section id="method" class="method-section">
        <p class="eyebrow">SYSTEM WORKFLOW</p>
        <h2>Từ dữ liệu thô đến quyết định có thể giải thích.</h2>
        <div class="method-grid">
          <article>
            <span>01</span><h3>Data Lake</h3>
            <p>SDSN CSV được hợp nhất, kiểm tra chất lượng và lưu dạng Parquet.</p>
          </article>
          <article>
            <span>02</span><h3>Machine Learning</h3>
            <p>Spark MLlib học quan hệ giữa các chỉ số và điểm Goal16.</p>
          </article>
          <article>
            <span>03</span><h3>Explainability</h3>
            <p>Hệ số và contribution chỉ ra yếu tố đang kéo điểm lên hoặc xuống.</p>
          </article>
          <article>
            <span>04</span><h3>Policy AI</h3>
            <p>RAG và LLM chuyển bằng chứng định lượng thành khuyến nghị dễ hiểu.</p>
          </article>
        </div>
      </section>
    </main>

    <footer>
      <div class="brand">
        <span class="brand-mark">16</span>
        <span><strong>SDG Intelligence</strong><small>Research prototype</small></span>
      </div>
      <p>Vue · Spring Boot · FastAPI · Spark · Qdrant</p>
    </footer>
  </div>
</template>
