import axios from "axios";
import { useQuery } from "@tanstack/react-query";

import { FINNHUB_API_KEY } from "../../apiKeys";

export const fetchSymbols = async () => {
    return await axios.get("https://finnhub.io/api/v1/stock/symbol", {
        params: {
            token: FINNHUB_API_KEY,
            function: "TIME_SERIES_DAILY",
            exchange: "US"
        }
    });
}

export const useSymbolsQuery = () => {
    return useQuery(["finnhub", "stock symbols"], () => fetchSymbols(), {
        cacheTime: 3600,
    });
};
