package vn.edu.sdg16.api.dto;

import java.util.Map;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

public record AskRequest(
        @NotBlank String question,
        @NotBlank String country,
        @NotNull @Min(2000) @Max(2100) Integer year,
        Map<String, @NotNull Double> features
) {
}
