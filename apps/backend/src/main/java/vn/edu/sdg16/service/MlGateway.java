package vn.edu.sdg16.service;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import org.springframework.stereotype.Service;
import org.springframework.http.MediaType;
import org.springframework.web.client.RestClient;

import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.JsonNode;
import vn.edu.sdg16.api.dto.AskRequest;
import vn.edu.sdg16.api.dto.PredictionRequest;
import vn.edu.sdg16.api.dto.SearchRequest;
import vn.edu.sdg16.config.MlServiceProperties;

@Service
public class MlGateway {

    private final RestClient restClient;
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final String baseUrl;

    public MlGateway(
            RestClient mlRestClient,
            MlServiceProperties properties,
            ObjectMapper objectMapper
    ) {
        this.restClient = mlRestClient;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(properties.connectTimeout())
                .build();
        this.objectMapper = objectMapper;
        this.baseUrl = properties.baseUrl().replaceAll("/+$", "");
    }

    public JsonNode health() {
        return restClient.get()
                .uri("/health")
                .retrieve()
                .body(JsonNode.class);
    }

    public JsonNode modelInfo() {
        return restClient.get()
                .uri("/model/info")
                .retrieve()
                .body(JsonNode.class);
    }

    public JsonNode predict(PredictionRequest request) {
        return post("/predict", request);
    }

    public JsonNode explain(PredictionRequest request) {
        return post("/explain", request);
    }

    public JsonNode search(SearchRequest request) {
        return post("/search", request);
    }

    public JsonNode ask(AskRequest request) {
        return post("/ask", request);
    }

    public JsonNode finalInsight(String country, Integer year, boolean useLlm) {
        return restClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("/insights/final")
                        .queryParam("country", country)
                        .queryParamIfPresent("year", java.util.Optional.ofNullable(year))
                        .queryParam("use_llm", useLlm)
                        .build())
                .retrieve()
                .body(JsonNode.class);
    }

    private JsonNode post(String path, Object body) {
        String json = toJson(body);
        HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + path))
                .header("Content-Type", MediaType.APPLICATION_JSON_VALUE)
                .POST(HttpRequest.BodyPublishers.ofString(json, java.nio.charset.StandardCharsets.UTF_8))
                .build();
        try {
            HttpResponse<String> response = httpClient.send(
                    request,
                    HttpResponse.BodyHandlers.ofString(java.nio.charset.StandardCharsets.UTF_8)
            );
            if (response.statusCode() >= 400) {
                throw new IllegalStateException(
                        "ML service returned " + response.statusCode() + ": " + response.body()
                );
            }
            return objectMapper.readTree(response.body());
        } catch (IOException exception) {
            throw new IllegalStateException("Cannot call ML service: " + exception.getMessage(), exception);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while calling ML service.", exception);
        }
    }

    private String toJson(Object body) {
        if (body instanceof PredictionRequest request) {
            return """
                    {"country":%s,"year":%d,"features":%s}
                    """.formatted(
                    quote(request.country()),
                    request.year(),
                    mapToJson(request.features())
            ).trim();
        }
        if (body instanceof AskRequest request) {
            return """
                    {"question":%s,"country":%s,"year":%d,"features":%s}
                    """.formatted(
                    quote(request.question()),
                    quote(request.country()),
                    request.year(),
                    request.features() == null ? "null" : mapToJson(request.features())
            ).trim();
        }
        if (body instanceof SearchRequest request) {
            return """
                    {"query":%s,"limit":%d}
                    """.formatted(quote(request.query()), request.limit()).trim();
        }
        throw new IllegalArgumentException("Unsupported request body: " + body.getClass().getName());
    }

    private String mapToJson(java.util.Map<String, Double> values) {
        StringBuilder builder = new StringBuilder("{");
        boolean first = true;
        for (java.util.Map.Entry<String, Double> entry : values.entrySet()) {
            if (!first) {
                builder.append(",");
            }
            first = false;
            builder.append(quote(entry.getKey())).append(":").append(entry.getValue());
        }
        return builder.append("}").toString();
    }

    private String quote(String value) {
        if (value == null) {
            return "null";
        }
        return "\"" + value
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t") + "\"";
    }
}
