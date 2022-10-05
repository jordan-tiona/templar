import axios from "axios";
import { useQuery } from "@tanstack/react-query";

import { ALPHA_VANTAGE_API_KEY } from "../../apiKeys";

export const fetchHistorical = async (symbol?: string) => {
    return symbol ? await axios.get("https://www.alphavantage.co/query", {
        params: {
            apikey: ALPHA_VANTAGE_API_KEY,
            function: "TIME_SERIES_DAILY",
            outputsize: "full",
            symbol: symbol,
        }
    }) : { data: [] };
}

export const useHistoricalQuery = (symbol?: string) => {
    return useQuery(["alpha vantage", "historical", symbol], () => fetchHistorical(symbol));
};
