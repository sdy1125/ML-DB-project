package vn.edu.sdg16.service;

import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

import tools.jackson.databind.JsonNode;
import vn.edu.sdg16.api.dto.AskRequest;
import vn.edu.sdg16.api.dto.PredictionRequest;
import vn.edu.sdg16.api.dto.SearchRequest;

@Service
public class MlGateway {

    private final RestClient restClient;

    public MlGateway(RestClient mlRestClient) {
        this.restClient = mlRestClient;
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

    private JsonNode post(String path, Object body) {
        return restClient.post()
                .uri(path)
                .body(body)
                .retrieve()
                .body(JsonNode.class);
    }
}
