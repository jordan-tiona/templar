import * as React from "react";
import { useRouteError } from "react-router-dom";

export type ErrorMessage = {
    data?: any,
    status?: number,
    statusText?: string,
    message?: string,
}

export const ErrorPage = () => {
    const error = ( useRouteError() as ErrorMessage );
    console.log( error );

    return (
        <>
            <h1>Oops!</h1>
            <p>Sorry, an unexpected error has occurred.</p>
            <p>{error.statusText || error.message}</p>
        </>
    )
};
