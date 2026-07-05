package vn.edu.sdg16.api;

import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import vn.edu.sdg16.api.dto.AskRequest;
import vn.edu.sdg16.api.dto.PredictionRequest;
import vn.edu.sdg16.api.dto.SearchRequest;
import vn.edu.sdg16.service.MlGateway;
import tools.jackson.databind.JsonNode;

@RestController
@RequestMapping("/api/v1")
public class AnalysisController {

    private final MlGateway mlGateway;

    public AnalysisController(MlGateway mlGateway) {
        this.mlGateway = mlGateway;
    }

    @GetMapping("/health")
    public JsonNode health() {
        return mlGateway.health();
    }

    @GetMapping("/model")
    public JsonNode modelInfo() {
        return mlGateway.modelInfo();
    }

    @PostMapping("/predictions")
    public JsonNode predict(@Valid @RequestBody PredictionRequest request) {
        return mlGateway.predict(request);
    }

    @PostMapping("/explanations")
    public JsonNode explain(@Valid @RequestBody PredictionRequest request) {
        return mlGateway.explain(request);
    }

    @PostMapping("/knowledge/search")
    public JsonNode search(@Valid @RequestBody SearchRequest request) {
        return mlGateway.search(request);
    }

    @PostMapping("/assistant/questions")
    public JsonNode ask(@Valid @RequestBody AskRequest request) {
        return mlGateway.ask(request);
    }

    @GetMapping("/insights/final")
    public JsonNode finalInsight(
            @RequestParam(defaultValue = "Vietnam") String country,
            @RequestParam(required = false) Integer year,
            @RequestParam(defaultValue = "true") boolean useLlm
    ) {
        return mlGateway.finalInsight(country, year, useLlm);
    }
}
