import * as React from "react";
import {
    EuiComboBox,
    EuiComboBoxOptionOption,
    EuiForm,
    EuiFormRow,
    EuiPageTemplate,
    EuiPanel,
} from "@elastic/eui";

import { useHistoricalQuery } from "../react-query/queries/alpha-vantage/historical";
import { useSymbolsQuery } from "../react-query/queries/finnhub/stock-symbols";

export const HomePage = () => {
    const symbols = useSymbolsQuery()?.data?.data;
    const [ selectedSymbol, setSelectedSymbol ] = React.useState<string>();
    const { data: timeSeriesRaw } = useHistoricalQuery( selectedSymbol );

    const symbolOptions: EuiComboBoxOptionOption[] = symbols?.map( ( symbol: { symbol: string } ) => ( { label: symbol.symbol } ) ) ?? [];

    return (
        <EuiPageTemplate
            panelled={false}
            restrictWidth
            grow
        >
            <EuiPageTemplate.Header
                pageTitle="Templar Swing Trader"
            />
            <EuiPageTemplate.Sidebar>
                <div>One Button</div>
            </EuiPageTemplate.Sidebar>
            <EuiPageTemplate.Section>
                <EuiPanel hasBorder={false} hasShadow={false}>
                    <EuiFormRow fullWidth label="Symbol:">
                        <EuiComboBox
                            fullWidth
                            onChange={( option ) => {
                                console.log( option );
                            }}
                            options={symbolOptions}
                            placeholder="Select symbol to view"
                            selectedOptions={ symbolOptions.filter( (option) => option.label === selectedSymbol )}
                            singleSelection={{ asPlainText: true }}
                        />
                    </EuiFormRow>
                </EuiPanel>
            </EuiPageTemplate.Section>
        </EuiPageTemplate>
    );
};
