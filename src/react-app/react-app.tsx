import * as React from "react";
import {
    QueryClient,
    QueryClientProvider,
} from "@tanstack/react-query";
import {
    EuiProvider,
} from "@elastic/eui";

import { Routes } from "../routes/routes";

export const App = () => {
    const queryClient = new QueryClient();

    return (
        <EuiProvider colorMode="dark">
            <QueryClientProvider client={queryClient}>
                <Routes />
            </QueryClientProvider>
        </EuiProvider>
    );
};
