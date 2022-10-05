import * as React from "react";
import {
    createBrowserRouter,
    RouterProvider,
} from "react-router-dom";

import { HomePage } from "./home";
import { ErrorPage } from "./error";

export const Routes = () => {

    const router = createBrowserRouter( [
        {
            path: "/",
            element: <HomePage />,
            errorElement: <ErrorPage />
        },
    ] );

    return (
        <RouterProvider router={router} />
    );
}